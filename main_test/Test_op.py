# from SNNCompiler.Compiler40nm import Compiler40nmSNN
from addict import Dict as AttrDict
import math
import brainpy as bp
from SNNCompiler.snn_compiler_64bit.backend.smt_64_op import OPType, RD_OPType, RS_OPType, ALU_OPType, ALUOUT_OPType
from SNNCompiler.snn_compiler_64bit.backend.smt_64_op import SMT_ASSIGN_I, SMT_RW, SMT_RC, SMT_IC
from SNNCompiler.snn_compiler_64bit.common.smt_64_reg import RegisterCollection
from SNNCompiler.snn_compiler_64bit.flow.smt_64_compiler import SMT64Compiler

def test():
    # region
    const = AttrDict()
    const.trst = 5
    const.vth = -50.0
    const.vrst = -65.0
    const.E1 = 0.0
    const.E2 = -80.0
    const.W_decay1  = math.exp(-1/5)
    const.W_decay1  = math.exp(-1/10)
    const.e1 = math.exp(-1/2)
    const.e2 = math.exp(-1/40)
    const.e3 = math.exp(-1/10)
    const.e4 = math.exp(-1/50)
    const.nv1 = 1.0
    const.nv2 = 2.0
    const.nv3 = 3.0
    const.nv4 = 4.0
    const.nv5 = 5.0
    const.nv6 = 6.0
    # endregion
    
    funcs = {"delta_V": bp.neurons.LIF(
            256,
            V_rest=-52,
            V_th=-67,
            V_reset=-22,
            tau=31,
            tau_ref=77,
            method="exp_auto",
            V_initializer=bp.init.Normal(-55.0, 2.0),).derivative}
    i_func = {
        "I": lambda g1, g2, V: 1.0 * g1 * (0 - V) + 1.0 * g2 * (-80.0 - V),
        }
    g_func = {
        "g1": lambda g1: g1 * math.exp(-1 / 5),
        "g2": lambda g2: g2 * math.exp(-1 / 10),
    }
    funcs.update(g_func)
    funcs.update(i_func)
    
    predefined_regs = {
        "V": "R2",
        "g1": "R4",
        "g2": "R5",
        "I": "R6",
    }
    constants = {"V_th": -55.0,"I_x": 3.0}

    _, v_compiler, statements = SMT64Compiler.compile_all(funcs=funcs, 
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
    
    
    flag = 1


if __name__ == "__main__":
    
    test()
    # result = SMT_ASSIGN_I(OP_TYPE=OPType.ASSIGN_IMM, IMM=1.0, NCU=1)
    # result = SMT_RW(OP_TYPE=OPType.NOP)
    # result = SMT_RC(OP_TYPE=OPType.CALCU_REG)
    # print(result.all_bin_fields)
    # print(result.dec_fields)
    # print(result.fields)
    # print(result.bin_fields)
    # print(result.bin_value)

    
    
    
    flag = 1