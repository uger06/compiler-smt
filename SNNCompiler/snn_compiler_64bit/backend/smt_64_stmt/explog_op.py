"""SMT 64位 指数对数 指令"""
# 2024.07@uger
from __future__ import annotations

import re
from typing import Union

import pyparsing as pp
from pyparsing import Group, Literal, Optional, Suppress, nested_expr, Keyword

from ...common.asm_IEEE754 import IEEE754
from ...common.smt_64_reg import Register64, RegisterCollection
from ..smt_64_op import CalField, OPType, ALU_OPType, RS_OPType, ALUOUT_OPType
from .smt64 import SMT64, SMT64ASMParseError


EXPLOG_OP_CODES: Union[OPType, ALU_OPType] = {
    "exp-r": (OPType.CALCU_REG, ALU_OPType.EXP_OP),  # exp(寄存器)
    "exp-i": (OPType.CALCU_IMM, ALU_OPType.EXP_OP),  # exp(立即数)
    "log-r": (OPType.CALCU_REG, ALU_OPType.LN_OP),  # log(寄存器)
    "log-i": (OPType.CALCU_IMM, ALU_OPType.LN_OP),  # log(立即数)
}
"""指数对数 表达式操作代码

- exp-r: exp(寄存器)
- exp-i: exp(立即数)
- log-r: log(寄存器)
- log-i: log(立即数)

"""

class EXPLOG_OP(SMT64):
    """运算指令

    - OPType.CALCU_REG, ALU_OPType.EXP_OP: exp(寄存器)
    - OPType.CALCU_IMM, ALU_OPType.EXP_OP: exp(立即数)
    - OPType.CALCU_REG, ALU_OPType.LN_OP): log(寄存器)
    - OPType.CALCU_IMM, ALU_OPType.LN_OP): log(立即数)

    Example:
        >>> regs = RegisterCollection()
        >>> IEEE754(-4.0)
        <IEEE754:3229614080:-4.00>
        >>> EXPLOG_OP("R5 = exp(R4)", regs)
        R5 = exp (R4)
        >>> EXPLOG_OP("R6 = log(R3)", regs)
        R6 = log (R3)
        
    NOTE: exp(src), 经由IR解析的src是一个处理好的表达式, 不存在负值
    """

    def __init__(self, expr: str, regs: RegisterCollection) -> None:
        """构造函数, 从表达式加载 `EXPLOG_OP` 对象, 只取第一个表达式

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合
        """
        super().__init__()

        if not expr:  # empty object
            return

        expr = re.split(r";|\n", expr)[0]  # 只取第一个表达式
        dst, op, a = self.parse_expr(expr, regs)[0]
        """
        [<R5 = 0, used by: []>, ['exp'], [<R4 = 0, used by: []>]]
        [<R7 = 0, used by: []>, ['-', 'exp'], [<IEEE754:1120403456:100.00>]]
        [<R6 = 0, used by: []>, ['exp'], ['-', <R3 = 0, used by: []>]]
        """

        a, optype, alu_code = self.preprocess_expr(a, op, expr)

        # self.rs_type = rs_type  # NOTE， 进入指数运算的已经是一个正操作数的表达式
        self.op_type = optype
        
        if optype == OPType.CALCU_REG:
            """
            CalField(op_code=OPType.CALCU_REG, fields=[0,10, 0,10, 0,6, 0,10, 0,10, 0,9])
            """
            part = (RS_OPType.NCU_ER_P.dec_value, a, RS_OPType.NCU_ER_P.dec_value, a, RS_OPType.NCU_ER_P.dec_value, dst, 
                    RS_OPType.ALU_OUT.dec_value, ALUOUT_OPType.alu_none.dec_value,   
                    RS_OPType.ALU_OUT.dec_value, ALUOUT_OPType.alu_none.dec_value, RS_OPType.NCU_ER_P.dec_value, 0)

            self.field = CalField(op_code=self.op_type, fields=part)

            self.op_0 = alu_code
            self.op_1 = ALU_OPType.NOP_OP
            
            self.rs0 = a
            self.rs1 = a
            self.rd = dst
            
        elif optype == OPType.CALCU_IMM:
            """
            CalField(op_code=OPType.CALCU_IMM, fields=[21.3, 0,10, 0,6, 0,9, 0,7])
            """
            part = (a, 
                    RS_OPType.ALU_OUT.dec_value, ALUOUT_OPType.alu_none.dec_value, RS_OPType.NCU_ER_P.dec_value, dst,   
                    RS_OPType.ALU_OUT.dec_value, ALUOUT_OPType.alu_none.dec_value, RS_OPType.NCU_ER_P.dec_value, 0)
            
            self.field = CalField(op_code=self.op_type, fields=part)

            self.op_0 = alu_code
            self.op_1 = ALU_OPType.NOP_OP
            
            self.rs0 = a
            self.rs1 = a
            self.rd = dst
             
    @classmethod
    def preprocess_expr(
        cls,
        a: list[Union[Register64, int, IEEE754]],
        op: str,
        expr: str,
    ) -> tuple[Union[int, IEEE754], OPType, ALU_OPType, RS_OPType, int]:
        """预处理表达式

        - 变换操作数顺序
        - 常数转换为 `IEEE754` 对象
        - 寄存器编号提取

        Example:
            >>> EXPLOG_OP.preprocess_expr([<R4 = 0, used by: []>], ['exp'], "R5 = exp(R4)")
            (<R4 = 0, used by: []>, <OPType.CALCU_REG: 'CALCU_REG'>, <ALU_OPType.EXP_OP: 'EXP_OP'>)
            >>> EXPLOG_OP.preprocess_expr([<IEEE754:1120403456:100.00>], ['log'], "R8 = log(100)")
            (<IEEE754:1120403456:100.00>, <OPType.CALCU_IMM: 'CALCU_IMM'>, <ALU_OPType.LN_OP: 'LN_OP'>)

        Args:
            a (list[str, Union[Register64, int, IEEE754]]): 操作数 a
            op (str): 操作符
            b (list[str, Union[Register64, int, IEEE754]]): 操作数 b
            expr (str): 表达式, 用于报错

        Returns:
            tuple[Union[int, IEEE754], OPCode, int]: 常数或寄存器编号, 操作符, 寄存器编号
        """
        fp_type = ""
        fp_type += "exp" if op[-1] == "exp" else "log"
        fp_type += "-"
        fp_type += "r" if isinstance(a[-1], Register64) else "i"
        
        optype, alu_code = EXPLOG_OP_CODES[fp_type]

        if isinstance(a[-1], (int, IEEE754)):  # 常数数值统一转换成 IEEE754 构造函数中处理
            a = IEEE754(a[-1] if a[0] != "-" else -a[-1])
        elif isinstance(a[-1], Register64):  # 寄存器编号
            a = a[-1]
        else:
            raise SMT64ASMParseError(f"不支持的 EXPLOG_OP 操作 {expr = }, {a = }")

        return a, optype, alu_code

    @classmethod
    def parse_expr(cls, expr: str, regs: RegisterCollection) -> list[list[Union[str, int]]]:
        """解析表达式为 dst, op, a 型式.

        - 浮点数常数转换成 `IEEE754` 对象
        - 整数常数被看作为浮点数转换成 `IEEE754` 对象
        - 常数和寄存器的符号提取出来, 之后变换表达式的时候需要用到
        - R* 直接转换成寄存器对象

        Example:
            >>> regs = RegisterCollection()
            >>> EXPLOG_OP.parse_expr("R7 = - exp(100.0)", regs)
            [<R7 = 0, used by: []>, ['-', 'exp'], [<IEEE754:1120403456:100.00>]]
            >>> EXPLOG_OP.parse_expr("R5 = exp(R4)", regs)
            [<R5 = 0, used by: []>, ['exp'], [<R4 = 0, used by: []>]]
            >>> EXPLOG_OP.parse_expr("R6 = exp(-R3)", regs)
            [<R6 = 0, used by: []>, ['exp'], ['-', <R3 = 0, used by: []>]]
            >>> e = EXPLOG_OP.parse_expr("R8 = log(R1)", regs)
            [<R8 = 0, used by: []>, ['log'], [<R1 = 0, used by: []>]]
            >>> EXPLOG_OP.parse_expr("R10 = -log(-100.0)", regs)
            [<R10 = 0, used by: []>, ['-', 'log'], ['-', <IEEE754:1120403456:100.00>]]      

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合

        Returns:
            list[list[Union[str, int]]]: 解析后的指令列表, `(src, dst)` 列表
        """

        constant = SMT64.get_constant_parser()
        register = SMT64.get_register_parser(regs)
        reg_or_constant = register | constant
        sign = Literal("+") | Literal("-")

        op = Group(Optional(sign) + Keyword("exp")) | Group(Optional(sign) + Keyword("log"))
        dst = register
        a = (
            nested_expr("(", ")", content=Optional(sign) + reg_or_constant)
        )

        assignment = dst + Suppress("=") + op + a

        result = []
        try:
            for one_expr in re.split(r";|\n", expr):
                if not (one_expr := one_expr.strip()):
                    continue
                dst, op, a = assignment.parse_string(one_expr, parse_all=True).as_list()
                # 数值转换成 IEEE754 对象,
                if isinstance(a[-1], (float, int)):
                    a[-1] = IEEE754(float(a[-1]))
                result.append([dst, op, a])
            return result
        except pp.ParseException as exc:
            raise SMT64ASMParseError(f"不支持的 EXPLOG_OP 操作 {one_expr}") from exc

    @property
    def asm_value(self) -> str:
        """汇编语句

        Example:
            >>> regs = RegisterCollection()
            >>> program = '''
            ... R5 = exp(R4)
            ... R8 = log(R1)
            ... R10 = log(-100.0)
            ... R7 = exp(50.0)
            ... '''
            >>> for op in EXPLOG_OP.create_from_expr(program, regs):
            ...     op.asm_value
            'R5 = exp (R4)'
            'R8 = log (R1)'
            'R9 = log (R0)'
            'R10 = log (-100.000000000000000000000000000000000000000000000000000000000000)'
            'R6 = exp (R3)'
            'R7 = exp (50.000000000000000000000000000000000000000000000000000000000000)'
            >>> program_code = "; ".join(op.asm_value for op in EXPLOG_OP.create_from_expr(program, regs))
            >>> program_code_code = "; ".join(op.asm_value for op in EXPLOG_OP.create_from_expr(program_code, regs))
            >>> program_code_code == program_code
            True

        Returns:
            str: 汇编语句
        """
        asm = []

        if self.field:
            a = SMT64.get_asm_value(self.rs0)
            b = SMT64.get_asm_value(self.rs1)
            dst = SMT64.get_asm_value(self.rd)

            sign = {
                (ALU_OPType.EXP_OP): "exp", 
                (ALU_OPType.LN_OP): "log",
            }[(self.op_0)]
            
            asm.append(
                {
                    OPType.CALCU_REG: f"{dst} = {sign} ({a})",
                    OPType.CALCU_IMM: f"{dst} = {sign} ({a})",
                }[self.op_type]
            )

        if not asm:
            raise ValueError("没有 EXPLOG_OP 操作")

        return ", ".join(asm)


__all__ = ["EXPLOG_OP"]

