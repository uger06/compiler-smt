# from SNNCompiler.snn_compiler_96bit.backend.smt_96_op import OPCode, OPField, OPType
# from SNNCompiler.snn_compiler_96bit.backend.smt_96_stmt.fp_op import FP_OP
# from SNNCompiler.snn_compiler_96bit.common.smt_96_reg import Register96, RegisterCollection

from SNNCompiler.snn_compiler_64bit.backend.smt_64_op import CalField, OPType, RD_OPType, RS_OPType, ALU_OPType
from SNNCompiler.snn_compiler_64bit.backend.smt_64_stmt.smt64 import SMT64

from SNNCompiler.snn_compiler_64bit.common.smt_64_reg import  RegisterCollection
from SNNCompiler.snn_compiler_64bit.backend.smt_64_stmt.cal_op import CAL_OP 
from SNNCompiler.snn_compiler_64bit.backend.smt_64_stmt.explog_op import EXPLOG_OP 

from SNNCompiler.snn_compiler_64bit.backend.smt_64_stmt.nop import NOP 

a = CalField(op_code = OPType.NOP)

regs = RegisterCollection()

program = '''
        R5 = exp(R4)
        R8 = log(R1)
        R9 = log(-R0)
        R10 = -log(-100.0)
        R6 = -exp(-R3)
        R7 = exp(50.0)
'''

program = '''
        R2 = R3 - 1.0
        R5 = R5 * R4
'''

a = CAL_OP("R3 = R2 - 5", regs)


for op in EXPLOG_OP.create_from_expr(program, regs):
    print(op.asm_value)

# b = EXPLOG_OP.create_from_expr(program, regs)

b = EXPLOG_OP.parse_expr("R7 = - exp(100.0)", regs)
c = EXPLOG_OP.parse_expr("R5 = exp(R4)", regs)
d = EXPLOG_OP.parse_expr("R6 = exp(-R3)", regs)
e = EXPLOG_OP.parse_expr("R8 = log(R1)", regs)
h = EXPLOG_OP.parse_expr("R10 = -log(-100.0)", regs)


b = NOP("NOP", regs)

a = CAL_OP("R5 = -4 + R4", regs)

b = CAL_OP("R5 = R7 + R8", regs)

op_type = OPType.CALCU_REG
field_all = CalField(op_code=op_type, fields=[0,10, 0,10, 0,6, 0,10, 0,10, 0,9])
bin_v = SMT64(op_type = op_type, op_0 = ALU_OPType.ADD_OP, op_1=ALU_OPType.ADD_OP, field = field_all)
print(bin_v.bin_value_for_smt)

op_type = OPType.CALCU_IMM
f2 = CalField(op_code=op_type, fields=[21.3, 0,10, 0,6, 0,9, 0,7])
bin_v = SMT64(op_type = op_type, op_0 = ALU_OPType.ADD_OP, op_1=ALU_OPType.ADD_OP, field = f2)
print(bin_v.bin_value_for_smt)

# f1 = CalField(op_code=OPType.CALCU_REG, fields=[0,10, 0,10, 6, 0,10, 0,10, 9])


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