import math
from addict import Dict as AttrDict
import brainpy as bp
import brainpy.math as bm
from brainpy._src.integrators import JointEq
from brainpy._src.dyn.neurons.base import GradNeuDyn, NeuDyn    
from brainpy.neurons import LIF
from brainpy.synapses import Exponential

from .snn_compiler_64bit.flow.smt_64_compiler import SMT64Compiler
from .snn_compiler_64bit.backend.smt_64_op import OPType, RD_OPType, RS_OPType, ALU_OPType
from .snn_compiler_64bit.backend.smt_64_op import SMT_ASSIGN_I, SMT_RW, SMT_RC
from .snn_compiler_64bit.common.smt_64_reg import RegisterCollection

def get_v_func(model):
    for ds in model.nodes().subset(NeuDyn).values():
        return ds.integral.f

def get_v_variable(model):
    for ds in model.nodes().subset(NeuDyn).values():
        return ds.integral.variables

def checkAttr(proj):
    return hasattr(proj, "comm") and hasattr(proj, "syn")  and hasattr(proj, "pre") and hasattr(proj, "post")

class SMTBase:
    def __init__(self, model):
        self.model = model
        self.cv = AttrDict()
    
        self.cv["tw"]  = 5000000
        self.cv["step_max"] = 0
        self.cv["neu_num"]  = 1023
        self.cv["rate"] = 0
        
    
        #Assume E and tau for synapse only depend on pre
        index = 0
        for proj in self.model.nodes().subset(bp.Projection).values():
            if checkAttr(proj):
                if proj.post != self.model.I:
                    continue
                index += 1                
                self.cv[f"tau{index}"] = proj.syn.tau
                self.cv[f"E{index}"] = proj.out.E if hasattr(proj, "refs") else proj.output.E

        #Assume E and I have identical parameters & V_reset、V_th、tau_ref这三个参数必须要有
        for ds in self.model.nodes().subset(NeuDyn).values():
            self.cv["V_reset"] = getattr(ds, "V_reset", -65.)
            self.cv["V_th"] = getattr(ds, "V_th", -50.) - 0.000000000000001
            self.cv["tau_ref"] = getattr(ds, "tau_ref", 5.)
            self.cv["V_rest"] = getattr(ds, "V_rest", -65.)
            self.cv["tau"] = getattr(ds, "tau", 5.)
            self.cv["R"] = getattr(ds, "R", 1.)
        

        # only one
        self.v_func = get_v_func(self.model)
        self.variables = get_v_variable(self.model)
        self.remaining_params = [key for key in self.variables if key not in ('V', 't', 'I')]    #  "R6" ~ "R7" 
        
        self.i_func = {
            "I": lambda g1, g2, V: self.cv['R'] * g1 * (self.cv['E1'] - V) + self.cv['R'] * g2 * (self.cv['E2'] - V),
        }

        self.g_func = {
            "g1": lambda g1: g1 * math.exp(-1 / self.cv['tau1']),
            "g2": lambda g2: g2 * math.exp(-1 / self.cv['tau2']),
        }

        
class SMT64bit(SMTBase):
    def __init__(self, model):
        super().__init__(model)
        
        self.model = model

    def func_to_64bit_smt(self,):
        
        # region
        const = AttrDict()
        const.W_decay1  = math.exp(-1/5)
        const.W_decay2  = math.exp(-1/10)
        const.RC_decay  = 1/20
        const.vu4 = -100
        0
        const.gu1 = 0.0
        const.gu2 = 0.0
        const.gu3 = 0.0
        const.gu4 = 0.0        
        const.e1 = math.exp(-1/2)
        const.e2 = math.exp(-1/40)
        const.e3 = math.exp(-1/10)
        const.e4 = math.exp(-1/50)
        const.E1 = 0.0
        const.E2 = -80.0
        const.vrst = -60.0
        const.trst = 5
        # endregion
        
        func = {"delta_V": self.v_func}
        func.update(self.g_func)
        # func.update(self.i_func)
        
        predefined_regs = {
            "V": "R2",
            "g1": "R4",
            "g2": "R5",
            "I": "R6",
        }
        constants = {"V_th": -55.0,"I_x": 3.0}
        
        _, v_compiler, statements = SMT64Compiler.compile_all(funcs=func,
                                                              constants=constants, 
                                                              predefined_regs=predefined_regs)

        const_list = []
        for _, value in const.items():
            const_list.append(value)
        
        smt_result = []
        smt_register = RegisterCollection()
        
        smt_register.regs[2].as_arg = 'LIF_V'
        smt_register.regs[4].as_arg = 'LIF_g1'
        smt_register.regs[5].as_arg = 'LIF_g2'
        smt_register.regs[10].as_arg = 'LIF_sum_w0'
        smt_register.regs[11].as_arg = 'LIF_sum_w1'
        
        # PART 1: NOP
        smt_result.append(SMT_RW(OP_TYPE=OPType.NOP))
        # PART 2: NCU_SR init
        ncu = 1
        while ncu <= 8:
            for i in range(16):
                instr = SMT_ASSIGN_I(OP_TYPE=OPType.ASSIGN_IMM, 
                                    NCU=ncu, 
                                    IMM=const_list[i], 
                                    RD_REG_0=smt_register.SR_regs[i], 
                                    RD_REG_1=smt_register.SR_regs[i])
                smt_result.append(instr)
            ncu+=1
        # PART 3: NCU_ER init
        ncu = 1
        imm = 146
        while ncu <= 8:
            for i in range(16):
                instr = SMT_ASSIGN_I(OP_TYPE=OPType.ASSIGN_IMM, 
                                    NCU=ncu, 
                                    IMM=imm, 
                                    RD_REG_0=smt_register.regs[i], 
                                    RD_REG_1=smt_register.regs[i])
                smt_result.append(instr)
                imm+=1
            ncu+=1
        # PART 4: syn_calcu
        tmp_len= len(smt_result)
        smt_result.append(SMT_RW(OP_TYPE=OPType.BUS_LOAD))
        smt_result.append(SMT_RW(OP_TYPE=OPType.SRAM_LOAD))
        smt_result.extend([SMT_RW(OP_TYPE=OPType.NOP) for _ in range(3)])
        smt_result.append(SMT_RC(OP_TYPE=OPType.CALCU_REG, 
                                ALU1_OP = ALU_OPType.ADD_OP,
                                RS1_OP_0 = RS_OPType.NCU_ER_P,
                                RS1_REG_0 = smt_register.regs[4],
                                RS1_OP_1  = RS_OPType.NCU_ER_P,
                                RS1_REG_1 = smt_register.regs[10],
                                RD1_OP  = RD_OPType.NCU_ER_RD_P,
                                RD1_REG = smt_register.regs[4],
                                ALU2_OP = ALU_OPType.ADD_OP,
                                RS2_OP_0 = RS_OPType.NCU_ER_P,
                                RS2_REG_0 = smt_register.regs[5],
                                RS2_OP_1  = RS_OPType.NCU_ER_P,
                                RS2_REG_1 = smt_register.regs[11],
                                RD2_OP   = RD_OPType.NCU_ER_RD_P,
                                RD2_REG = smt_register.regs[5]))
        smt_result.extend([SMT_RW(OP_TYPE=OPType.NOP) for _ in range(5)])
        smt_result.append(SMT_RW(OP_TYPE=OPType.SRAM_SAVE))
        tmp_len= len(smt_result)
        smt_result.extend([SMT_RW(OP_TYPE=OPType.NOP) for _ in range(385-tmp_len)])
        
        # PART 5: neu_calcu
        smt_result.append(SMT_RW(OP_TYPE=OPType.SRAM_LOAD))
        smt_result.extend([SMT_RW(OP_TYPE=OPType.NOP) for _ in range(3)])
        """smt begin"""
        smt_result.extend(statements)
        smt_result.append(SMT_RC(OP_TYPE=OPType.VSET, 
                                ALU1_OP = ALU_OPType.ENABLE_OP, 
                                RS1_OP_0 = RS_OPType.NCU_SR_P,
                                RS1_REG_0 = smt_register.SR_regs[14],
                                RS1_OP_1  = RS_OPType.NCU_SR_P,
                                RS1_REG_1= smt_register.SR_regs[14],
                                RD1_OP  = RD_OPType.NCU_ER_RD_P,
                                RD1_REG= RegisterCollection().regs[2]))
        smt_result.append(SMT_RW(OP_TYPE=OPType.SPIKE_GEN))
        smt_result.append(SMT_RW(OP_TYPE=OPType.NOP))
        smt_result.append(SMT_RW(OP_TYPE=OPType.SRAM_SAVE))
        tmp_len= len(smt_result)
        if tmp_len < 513:
            smt_result.extend([SMT_RW(OP_TYPE=OPType.NOP) for _ in range(513-tmp_len)])
        """smt end"""
        tmp_len= len(smt_result)
        # PART 6: syn_update_calcu:
        smt_result.extend([SMT_RW(OP_TYPE=OPType.NOP) for _ in range(128)])
        # PART 7: gemm_calcu
        smt_result.extend([SMT_RW(OP_TYPE=OPType.NOP) for _ in range(128)])
                    
        return smt_result
