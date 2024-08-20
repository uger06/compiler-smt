from SNNCompiler.snn_compiler_96bit.backend.smt_96_op import OPCode, OPField, OPType
from SNNCompiler.snn_compiler_96bit.backend.smt_96_stmt.fp_op import FP_OP
from SNNCompiler.snn_compiler_96bit.backend.smt_96_stmt.smt96 import SMT96
from SNNCompiler.snn_compiler_96bit.common.smt_96_reg import Register96, RegisterCollection
from SNNCompiler.snn_compiler_96bit.flow.smt_96_compiler import SMT96Compiler

from SNNCompiler.snn_compiler_64bit.backend.smt_64_op import CalField, OPType, RD_OPType, RS_OPType, ALU_OPType
# from SNNCompiler.snn_compiler_64bit.backend.smt_64_stmt.smt64 import SMT64

import brainpy as bp
funcs = {"V": bp.neurons.LIF(256).derivative}
compiler = SMT96Compiler(funcs=funcs)
compiler.update_stablehlo_statements()


funcs = {"V": bp.neurons.LIF(
            256,
            V_rest=-52,
            V_th=-67,
            V_reset=-22,
            tau=31,
            tau_ref=77,
            method="exp_auto",
            V_initializer=bp.init.Normal(-55.0, 2.0),).derivative}

constants = {"V_th": 2.0, # 硬件ASIC输入的常数
            "I_x": 3.0}

predefined_regs = {
    # "zero0": "R0",
    "V": "R3",
}


_, v_compiler, statements = SMT96Compiler.compile_all(funcs=funcs, 
                                                                constants=constants,
                                                                predefined_regs=predefined_regs)

flag = 1



regs = RegisterCollection()
a = SMT96.get_constant_parser().parse_string("WRIGHT_RX_READY").as_list()
b = SMT96.get_constant_parser().parse_string("CTRL_PULSE.WRIGHT_RX_READY").as_list()

op_type = OPType.NOP
field_all = OPField(op_code=OPCode.NOP, fields=[])
SMT96(op_type, OPCode.NOP, field_all, OPCode.NOP, field_all).bin_value_for_human



# f1 = CalField(op_code=OPType.CALCU_REG, fields=[0,10, 0,10, 6, 0,10, 0,10, 9])
# f2 = CalField(op_code=OPType.CALCU_IMM, fields=[21.3, 0,10, 6, 0,9, 7])

# f = OPField(op_code=OPCode.REG_ADD_POS, fields=[10, 5, 1])

# regs = RegisterCollection()
# a = FP_OP("R2 = -4 - R3", regs)
# b = FP_OP("R2 = -4.0 + R3", regs)

# regs = RegisterCollection()
# program = """
#     R2 = R3 - 1.0
#     R4 = R10 * R11
#     """
# c = list(FP_OP.create_from_expr(program, regs))



"""
根据返回的fp_type, 确定ncu_er_p 还是ncu_er_n? 

    action, op_code = EXP_OP_CODES[fp_type]

EXP_OP_CODES: dict[str, tuple[str, Union[OPCode, None]]] = {
    "+i++r": ("", OPCode.IMM_ADD_POS),  # 数值加寄存器
    "+i+-r": ("", OPCode.IMM_ADD_NEG),  # 数值减寄存器
    "+i-+r": ("", OPCode.IMM_ADD_NEG),  # 数值减寄存器
    "+i--r": ("", OPCode.IMM_ADD_POS),  # 数值加寄存器
    "+i*+r": ("", OPCode.IMM_MUL_POS),  # 数值乘寄存器

"""
# print(f)