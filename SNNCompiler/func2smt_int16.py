import math
from addict import Dict as AttrDict
import brainpy as bp
import brainpy.math as bm
from brainpy._src.integrators import JointEq
from brainpy._src.dyn.neurons.base import GradNeuDyn, NeuDyn    
from brainpy.neurons import LIF
from brainpy.synapses import Exponential
import numpy as np

# FIXME: modify the import package name
from .snn_compiler_32bit.flow.smt_compiler import SMTCompiler
from .snn_compiler_96bit.flow.smt_96_compiler import SMT96Compiler
from .snn_compiler_96bit.common.smt_96_base import CTRL_LEVEL, CTRL_PULSE, IBinary, IEEE754
from .snn_compiler_96bit.backend.smt_96_stmt.smt96 import SMT96

def get_v_func(model):
    for ds in model.nodes().subset(NeuDyn).values():
        return ds.integral.f

def get_v_variable(model):
    for ds in model.nodes().subset(NeuDyn).values():
        return ds.integral.variables

class SMTBase:
    def __init__(self, model):
        self.model = model
        self.cv = AttrDict()
    
        self.cv["tw"]  = 50000000
        self.cv["step_max"] = 0
        self.cv["neu_num"]  = 1023
        self.cv["rate"] = 0
        
        index = 0
        for proj in self.model.nodes().subset(bp.Projection).values():
            if hasattr(proj, "refs"):
                # NEW brainpy version: net = EINet(bp.DynSysGroup)
                if proj.refs["post"] != self.model.I:
                    continue
                index += 1
                self.cv[f"E{index}"] = proj.refs["out"].E
                self.cv[f"tau{index}"] = proj.refs["syn"].tau
            else:
                # OLD brainpy version: net = EINet_v1(bp.Network)
                # if proj.post != self.model.I:
                #     continue
                index += 1
                self.cv[f"E{index}"] = proj.output.E
                self.cv[f"tau{index}"] = proj.syn.tau
        
        #TODO: uger, maybe need to support more than 2 neurons parameters
        for ds in self.model.nodes().subset(NeuDyn).values():
            self.cv["V_reset"] = getattr(ds, "V_reset", -65.)
            self.cv["V_th"] = getattr(ds, "V_th", -50.)
            self.cv["V_rest"] = getattr(ds, "V_rest", -65.)
            self.cv["tau_ref"] = getattr(ds, "tau_ref", 5.)
            self.cv["tau"] = getattr(ds, "tau", 5.)
            self.cv["R"] = getattr(ds, "R", 1.)

        # only one
        # self.v_func = get_v_func(self.model) 
        # self.v_func_1 = get_v_func(self.model)
        # self.v_func_2 = get_v_func(self.model)

        # self.variables = get_v_variable(self.model)
        # self.remaining_params = [key for key in self.variables if key not in ('V', 't', 'I')]    #  "R6" ~ "R7" 
        
        # self.i_func = {
        #     "I": lambda g1, g2, V: self.cv['R'] * g1 * (self.cv['E1'] - V) + self.cv['R'] * g2 * (self.cv['E2'] - V),
        # }
        # self.g_func = {
        #     "g1": lambda g1: g1 * math.exp(-1 / self.cv['tau1']),
        #     "g2": lambda g2: g2 * math.exp(-1 / self.cv['tau2']),
        # }

        # self.i_func = {
        #     "I": lambda g0, V: self.cv['R'] * g0 * (self.cv['E1'] - V),
        # }
        
        # self.g_func = {
        #     "g0": lambda g0: g0 * math.exp(-1 / self.cv['tau1']),
        # }
        
        
class SMT32bit(SMTBase):
    def __init__(self, model):
        super().__init__(model)
        
        self.model = model
        self.result_bits = 3  # or 4
        self.cv["result_bits"] = self.result_bits

    def func_to_32bit_smt(self,):
        # FPGA版本
        func = {"V": self.v_func}
        func.update(self.g_func)

        predefined_regs = {
            "g1": "R2",
            "g2": "R3",
        }

        i_compiler, v_compiler, smt_result = SMTCompiler.compile_all(
            func=func,
            preload_constants=self.cv,
            predefined_regs=predefined_regs,
            i_func=self.i_func["I"],
            update_method={"I": "update", "g1": "update", "g2": "update"},
            result_bits=self.result_bits,
        )

        all_constants = i_compiler.preload_constants | v_compiler.preload_constants
        printed_name = []
        register_constants = []
        all_constants_tmp = sorted(all_constants, key=lambda r: int(r.name[2:]))
        for pc in all_constants_tmp:
            if pc.name not in printed_name:
                printed_name.append(pc.name)
                register_constants.append(pc)
                
        return smt_result, register_constants, self.cv

class SMT96bit(SMTBase):
    def __init__(self, model, config):
        super().__init__(model)
        self.model = model
        self.cv['neuron_num'] = config['neuron_num']
        self.cv['total_step'] = 0
        self.cv['I'] = 1.0
        self.cv['weight_phase'] = 1048576
        
        self.smt_result, self.v_compiler = self.func_to_96bit_smt_init()
        
    def func_to_96bit_smt_cus(self, npu_id):
        
        pre_func = f"""
            03: R_CHIP_NPU_ID = {npu_id}, R_NEU_NUMS = {self.cv['neuron_num']}
        """
             
        pre_func_stmts = list(SMT96.create_from_expr(pre_func, regs=self.v_compiler.regs))
        self.smt_result[2] = pre_func_stmts[0]
        

        import random
        if npu_id == random.choice(list(range(16))):
            import copy
            smt_result = copy.deepcopy(self.smt_result)
            
            Ix_dec = np.uint32((2<<16) + (2<<0))
            Ix = IEEE754.ieee754_to_float(Ix_dec)
            Ix = f'{Ix:.60f}'
            
            for r in self.v_compiler.regs.used_regs:
                if r.as_return == "V0":
                    D_V0 = r.name
                elif r.as_return == "V1":
                    D_V1 = r.name
                    
            post_func = f"""
                74: {D_V0} = {D_V0} + {Ix}
                75: {D_V1} = {D_V1} + {Ix}
            """
            
            post_func_stmts = list(SMT96.create_from_expr(post_func, regs=self.v_compiler.regs))
            smt_result[105:107] = post_func_stmts[0:]      ## note where to replace the post_func_stmts
            return smt_result
            
        return self.smt_result         


    def func_to_96bit_smt_init(self,):
        trst_set, vrst = 0, 0
        trst = np.uint32((trst_set<<16) + (trst_set<<0))
        vrst = np.uint32((vrst<<16) + (vrst<<0))

        tau_set = 1
        tau_dec = np.uint32((tau_set<<16) + (tau_set<<0))
        tau = IEEE754.ieee754_to_float(tau_dec)

        Ix_set = 0
        Ix_dec = np.uint32((Ix_set<<16) + (Ix_set<<0))
        Ix = IEEE754.ieee754_to_float(Ix_dec)

        Vth_set = 5
        Vth_dec = np.uint32((Vth_set<<16) + (Vth_set<<0))
        vth = IEEE754.ieee754_to_float(Vth_dec)
        vth = f'{vth:.60f}'

        W_decay1_set, E1_set = 1, 2
        W_decay1_dec = np.uint32((W_decay1_set<<16) + (W_decay1_set<<0))
        W_decay1     = IEEE754.ieee754_to_float(W_decay1_dec)
        E1_dec       = np.uint32((E1_set<<16) + (E1_set<<0))
        E1           = IEEE754.ieee754_to_float(E1_dec)
        

        fac_dec = np.uint32((1 << 16) + (1 << 0))
        fac_dec = IEEE754.ieee754_to_float(fac_dec)
        
        self.v_func_1 = {
            "V0": lambda V0, I0: (V0 * fac_dec + I0) * tau,
            "V1": lambda V1, I1: (V1 * fac_dec + I1) * tau,
        }


        self.g_func = {
            "g0": lambda g0: g0 * W_decay1,
            "g1": lambda g1: g1 * W_decay1,
        }
        
        self.i_func = {
            "I0": lambda g0, V0: g0 * (E1 - V0),
            "I1": lambda g1, V1: g1 * (E1 - V1),
        }
        
        funcs = self.v_func_1
        funcs.update(self.g_func)
        funcs.update(self.i_func)
        
        
        cfg = 2 #NOTE: for int16
        rnd = 0
        rate = 0
        total_num = self.cv['neuron_num']
        if total_num > 1023:
            neu_num = 1024
        else:
            neu_num = total_num

        # region
        neu_num = IBinary.dec2bin(neu_num, 11)
        cfg_bin = IBinary.dec2bin(cfg, 13 - 11)
        rnd_bin = IBinary.dec2bin(rnd, 17 - 13)
        rate_bin = IBinary.dec2bin(rate, 32 - 17)
        neu_num_bin = rate_bin + rnd_bin + cfg_bin + neu_num
        neu_num = IBinary(0, 32)
        neu_num.bin_value = neu_num_bin
        neu_num = neu_num.dec_value
        total_step = self.cv['total_step']
        
        self.cv['neuron_num'] = neu_num
        ndma_phase = 32800
        weight_phase = int(math.pow(2, 25))  # math.pow(2, 20), must be int
        constants = {"V_th": vth, # 硬件ASIC输入的常数
                     "I_x": Ix, 
                     "INT": 16}  
        # endregion
        
        predefined_regs = {
            "neu_en1":  "R0",
            "tlastsp1": "R1",
            "V1":       "R2",
            "g1":       "R3",
            "neu_en0":  "R4",
            "tlastsp0": "R5",
            "V0":       "R6",
            "g0":       "R7",
            # "w_en":     "R8",
            # "neu_id":   "R9",            
            "w1":       "R10",
            "w2":       "R11",
        }    
        
        
        _, v_compiler, smt_result = SMT96Compiler.compile_all(
            funcs=funcs,
            constants=constants,
            predefined_regs=predefined_regs,
            update_method={"g1": "update", "g2": "update"},
        )

        pre_func = f"""
            01: R_CTRL_LEVEL = CFG_EN, R_ZERO_REG = 0
            02: R_STEP_REG = {total_step}, R_PHASE = 0
            03: R_CHIP_NPU_ID = 0, R_NEU_NUMS = {neu_num}
            04: R_TRST_REG1 = {trst}, R_TRST_REG0 = {trst}  //uger debug #2.0
            05: R_VRST_REG1 = {vrst}, R_VRST_REG0 = {vrst}
            06: R_V_DIFF_REG1 = 0, R_V_DIFF_REG0 = 0
            07: R_TLASTSP_TMP1 = 0, R_TLASTSP_TMP0 = 0
            08: R_NONE_REG = 0, R_CTRL_PULSE = LFSR_INIT_SET //uger debug #2.1
            09: R_NONE_REG = 1, R_CTRL_PULSE = LFSR_SET //uger debug #2.2
            10: NOP
            11: NOP
            12: R_CTRL_LEVEL = NPU_RST_EN, R_NONE_REG = 0   //uger debug #2.3
            13: JUMP 0, 0, -1  // ndma rd wait
            14: NOP
            15: R_PHASE = {ndma_phase}, R_CTRL_PULSE = TIMER_SET
            16: R_CTRL_LEVEL = {CTRL_LEVEL.SIM_EN + CTRL_LEVEL.NDMA_RD_EN}, R_NONE_REG = 0 // sim_en on & ndma_en on // uger debug #2.4
            17: JUMP 0, 0, -1  // ndma rd wait
            18: NOP
            19: R_PHASE = {weight_phase}, R_CTRL_PULSE = TIMER_SET  // set wacc
            20: R_CTRL_LEVEL = {CTRL_LEVEL.SIM_EN + CTRL_LEVEL.W_EN}, R_NONE_REG = 0  // sim_en on & w_en on
            21: JUMP 13, -1, 1  // uger debug #1
            22: NOP
            23: R_w1 <= R_8, R_w2 <= R_9, R_CTRL_PULSE = WRIGHT_RX_READY // uger debug #2, debug #20240511
            24: NOP
            25: R_neu_en1 = SRAM[0], R_tlastsp1 = SRAM[1], R_V1 = SRAM[2], R_g1 = SRAM[3], R_neu_en0 = SRAM[4], R_tlastsp0 = SRAM[5], R_V0 = SRAM[6], R_g0 = SRAM[7]  // uger debug #3.1, #20240613
            26: R_g0 = R_w1 + R_g0
            27: R_g1 = R_w2 + R_g1
            28: NOP
            29: NOP
            30: NOP
            31: NOP
            32: SRAM[0] = R_neu_en1, SRAM[1] = R_tlastsp1, SRAM[2] = R_V1, SRAM[3] = R_g1, SRAM[4] = R_neu_en0, SRAM[5] = R_tlastsp0, SRAM[6] = R_V0, SRAM[7] = R_g0  // uger debug #3.2, #20240613
            33: JUMP({CTRL_PULSE.SMT_JUMP + CTRL_PULSE.W_JUMP}) 0, 0, -13
            34: NOP
            35: R_CTRL_LEVEL = {CTRL_LEVEL.SIM_EN + CTRL_LEVEL.V_EN}, R_CTRL_PULSE = NPU_SET
            36: JUMP 0, {2 + len(smt_result) + 3}, 1   //uger debug #3. FIX: jump to post_func: 86
            37: NOP
            38: R_neu_en1 = SRAM[0], R_TLASTSP_TMP1 = SRAM[1], R_V1 = SRAM[2], R_g1 = SRAM[3], R_neu_en0 = SRAM[4], R_TLASTSP_TMP0 = SRAM[5], R_V0 = SRAM[6], R_g0 = SRAM[7]  // uger debug #3.3, #20240613
        """

        pre_func_stmts = list(SMT96.create_from_expr(pre_func, regs=v_compiler.regs))

        post_func = f"""
            83: R_tlastsp0 = R_TLASTSP_TMP0, R_tlastsp1 = R_TLASTSP_TMP1, R_NONE_REG = 0  //uger debug #2.6
            84: SRAM[0] = R_neu_en1, SRAM[1] = R_tlastsp1, SRAM[2] = R_V1, SRAM[3] = R_g1, SRAM[4] = R_neu_en0, SRAM[5] = R_tlastsp0, SRAM[6] = R_V0, SRAM[7] = R_g0  // uger debug #3.4, #20240613
            85: JUMP({CTRL_PULSE.SMT_JUMP + CTRL_PULSE.NPU_PLUS}) 0, 0, {-(len(pre_func_stmts) + len(smt_result) + 2 + 1 - 35)}
            86: NOP
            87: R_CTRL_LEVEL = {CTRL_LEVEL.SIM_EN + CTRL_LEVEL.S_EN}, R_CTRL_PULSE = {CTRL_PULSE.NPU_SET}
            88: JUMP({CTRL_PULSE.SMT_JUMP + CTRL_PULSE.STEP_PLUS}) 0, {-(len(pre_func_stmts) + len(smt_result) + 5 + 1 - 18)}, 1
            89: NOP
            90: R_PHASE = {ndma_phase}, R_CTRL_PULSE = {CTRL_PULSE.TIMER_SET + CTRL_PULSE.NPU_SET}
            91: R_CTRL_LEVEL = {CTRL_LEVEL.SIM_EN + CTRL_LEVEL.NDMA_WR_EN}
            92: JUMP 0, 0, -1  //uger debug #4
            93: NOP
            94: R_CTRL_LEVEL = {CTRL_LEVEL.SIM_EN}
            95: JUMP 0, 0, {-(len(pre_func_stmts) + len(smt_result) + 12 + 1 - 11)}
            96: R_CTRL_LEVEL = {CTRL_LEVEL.NPU_RST_EN}, R_CTRL_PULSE = {CTRL_PULSE.SIM_END} //uger debug #5
        """

        post_func_stmts = list(SMT96.create_from_expr(post_func, regs=v_compiler.regs))

        nops = "\n"
        for i in range(256 - (len(pre_func_stmts) + len(smt_result) + len(post_func_stmts))):
            nops += f"{i}: NOP\n"

        nops = list(SMT96.create_from_expr(nops, regs=v_compiler.regs))

        self.smt_result = pre_func_stmts + smt_result + post_func_stmts + nops

        self.v_compiler = v_compiler

        return self.smt_result, self.v_compiler