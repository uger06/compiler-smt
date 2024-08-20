"""SMT 64位 加(减)乘除 指令"""
# 2024.07@uger
from __future__ import annotations

import re
from typing import Union

import pyparsing as pp
from pyparsing import Group, Literal, Optional, Suppress, nested_expr

from ...common.asm_IEEE754 import IEEE754
from ...common.smt_64_reg import Register64, RegisterCollection
from ..smt_64_op import CalField, OPType, ALU_OPType, RS_OPType, ALUOUT_OPType
from .smt64 import SMT64, SMT64ASMParseError


CAL_OP_CODES: dict[str, tuple[str, Union[OPType, ALU_OPType, RS_OPType, None]]] = {
    "+i++r": ("", OPType.CALCU_IMM, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_P),  # 数值加寄存器
    "+i+-r": ("", OPType.CALCU_IMM, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_N),  # 数值减寄存器
    "+i-+r": ("", OPType.CALCU_IMM, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_N),  # 数值减寄存器
    "+i--r": ("", OPType.CALCU_IMM, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_P),  # 数值加寄存器
    "+i*+r": ("", OPType.CALCU_IMM, ALU_OPType.MUL_OP, RS_OPType.NCU_ER_P),  # 数值乘寄存器
    "+i*-r": ("", OPType.CALCU_IMM, ALU_OPType.MUL_OP, RS_OPType.NCU_ER_N),  # 数值乘负寄存器
    "-i++r": ("", OPType.CALCU_IMM, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_P),  # 数值加寄存器
    "-i+-r": ("", OPType.CALCU_IMM, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_N),  # 数值减寄存器
    "-i-+r": ("", OPType.CALCU_IMM, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_N),  # 数值减寄存器
    "-i--r": ("", OPType.CALCU_IMM, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_P),  # 数值加寄存器
    "-i*+r": ("", OPType.CALCU_IMM, ALU_OPType.MUL_OP, RS_OPType.NCU_ER_P),  # 数值乘寄存器
    "-i*-r": ("", OPType.CALCU_IMM, ALU_OPType.MUL_OP, RS_OPType.NCU_ER_N),  # 数值乘负寄存器
    "+r++r": ("", OPType.CALCU_REG, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_P),  # 寄存器加寄存器
    "+r+-r": ("", OPType.CALCU_REG, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_N),  # 寄存器减寄存器
    "+r-+r": ("", OPType.CALCU_REG, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_N),  # 寄存器减寄存器
    "+r--r": ("", OPType.CALCU_REG, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_P),  # 寄存器加寄存器
    "+r*+r": ("", OPType.CALCU_REG, ALU_OPType.MUL_OP, RS_OPType.NCU_ER_P),  # 寄存器乘寄存器
    "+r*-r": ("", OPType.CALCU_REG, ALU_OPType.MUL_OP, RS_OPType.NCU_ER_N),  # 寄存器乘负寄存器
    "-r++r": ("swap", OPType.CALCU_REG, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_N),  # a b = b a 之后寄存器减寄存器
    "-r+-r": ("todo", None),
    "-r-+r": ("todo", None),
    "-r--r": ("swap", OPType.CALCU_REG, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_N),  # a b = b a 之后寄存器减寄存器
    "-r*+r": ("swap", OPType.CALCU_REG, ALU_OPType.MUL_OP, RS_OPType.NCU_ER_N),  # a b = b a 之后寄存器乘负寄存器
    "-r*-r": ("", OPType.CALCU_REG, ALU_OPType.MUL_OP, RS_OPType.NCU_ER_P),  # 寄存器乘寄存器
    "+r++i": ("swap", OPType.CALCU_IMM, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_P),  # a b = b a 之后数值加寄存器
    "-r++i": ("swap", OPType.CALCU_IMM, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_N),  # a b = b a 之后数值减寄存器
    "+r-+i": ("swap-", OPType.CALCU_IMM, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_P),  # a b = -b a 之后数值加寄存器
    "-r-+i": ("swap-", OPType.CALCU_IMM, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_N),  # a b = -b a 之后数值减寄存器
    "+r*+i": ("swap", OPType.CALCU_IMM, ALU_OPType.MUL_OP, RS_OPType.NCU_ER_P),  # a b = b a 之后数值乘寄存器
    "-r*+i": ("swap", OPType.CALCU_IMM, ALU_OPType.MUL_OP, RS_OPType.NCU_ER_N),  # a b = b a 之后数值乘负寄存器
    "+r+-i": ("swap", OPType.CALCU_IMM, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_P),  # a b = b a 之后数值加寄存器
    "-r+-i": ("swap", OPType.CALCU_IMM, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_N),  # a b = b a 之后数值减寄存器
    "+r--i": ("swap+", OPType.CALCU_IMM, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_P),  # a b = abs(b) a 之后数值加寄存器
    "-r--i": ("swap+", OPType.CALCU_IMM, ALU_OPType.ADD_OP, RS_OPType.NCU_ER_N),  # a b = abs(b) a 之后数值加寄存器
    "+r*-i": ("swap", OPType.CALCU_IMM, ALU_OPType.MUL_OP, RS_OPType.NCU_ER_P),  # a b = b a 之后数值乘寄存器
    "-r*-i": ("swap", OPType.CALCU_IMM, ALU_OPType.MUL_OP, RS_OPType.NCU_ER_N),  # a b = b a 之后数值乘负寄存器
    
    "+i/+r": ("", OPType.CALCU_IMM, ALU_OPType.DIV_OP, RS_OPType.NCU_ER_P),  # 数值除寄存器
    "+i/-r": ("", OPType.CALCU_IMM, ALU_OPType.DIV_OP, RS_OPType.NCU_ER_N),  # 数值除负寄存器
    "-i/+r": ("", OPType.CALCU_IMM, ALU_OPType.DIV_OP, RS_OPType.NCU_ER_P),  # 数值除寄存器
    "-i/-r": ("", OPType.CALCU_IMM, ALU_OPType.DIV_OP, RS_OPType.NCU_ER_N),  # 数值除负寄存器
    "+r/+r": ("", OPType.CALCU_REG, ALU_OPType.DIV_OP, RS_OPType.NCU_ER_P),  # 寄存器除寄存器
    "+r/-r": ("", OPType.CALCU_REG, ALU_OPType.DIV_OP, RS_OPType.NCU_ER_N),  # 寄存器除负寄存器
    "-r/+r": ("swap", OPType.CALCU_REG, ALU_OPType.DIV_OP, RS_OPType.NCU_ER_N),  # a b = b a 之后寄存器除负寄存器
    "-r/-r": ("todo", None),
}
"""加(减)乘除 表达式操作代码

- todo: 准备实现
- swap: a b = b a 之后实现
- swap-: a b = -b a 之后实现, 用于实现寄存器减数值
- swap+: a b = abs(b) a 之后实现, 用于实现寄存器减负数
- 其他: 直接实现
"""


class CAL_OP(SMT64):
    """运算指令

    - OPType.CALCU_REG, ALU_OPType.ADD_OP: 寄存器加寄存器
    - OPType.CALCU_REG, ALU_OPType.MUL_OP: 寄存器乘寄存器
    - OPType.CALCU_REG, ALU_OPType.DIV_OP: 寄存器除寄存器
    - OPType.CALCU_IMM, ALU_OPType.ADD_OP: 立即数加寄存器
    - OPType.CALCU_IMM, ALU_OPType.MUL_OP: 立即数乘寄存器
    - OPType.CALCU_IMM, ALU_OPType.DIV_OP: 立即数除寄存器

    Example:
        >>> regs = RegisterCollection()
        >>> CAL_OP("R3 = R2 - 5", regs)
        R3 = -5.00 + R2
        >>> CAL_OP("R5 = R7 + R8", regs)
        R5 = R7 + R8
        >>> regs = RegisterCollection()
        >>> program = '''
        ... R2 = R3 - 1.0
        ... R5 = R5 * R4
        ... '''
        >>> list(CAL_OP.create_from_expr(program, regs))
        [R2 = -1.00 + R3, R5 = R5 * R4]
    """

    def __init__(self, expr: str, regs: RegisterCollection) -> None:
        """构造函数, 从表达式加载 `CAL_OP` 对象, 只取第一个表达式

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合
        """
        super().__init__()

        if not expr:  # empty object
            return

        expr = re.split(r";|\n", expr)[0]  # 只取第一个表达式
        dst, a, op, b = self.parse_expr(expr, regs)[0]
        """
        (<R2 = 0, used by: []>, ['-', <IEEE754:1082130432:4.00>], '+', [<R3 = 0, used by: []>])
        """

        if isinstance(a[-1], IEEE754) and isinstance(b[-1], IEEE754):
            raise NotImplementedError(f"暂不支持纯常数运算 {expr =}")

        a, optype, alu_code, rs_type, b = self.preprocess_expr(a, op, b, expr)
        """
        (<IEEE754:3229614080:-4.00>, <OPCode.CALCU_IMM>, <ALU_OPType.ADD_OP>, <R3 = 0, used by: []>)
        """
        
        self.rs_type = rs_type
        self.op_type = optype

        if optype == OPType.CALCU_REG:
            """
            CalField(op_code=OPType.CALCU_REG, fields=[0,10, 0,10, 0,6, 0,10, 0,10, 0,9])
            """
            if op in "+-*/":
                part = (RS_OPType.NCU_ER_P.dec_value, a, rs_type.dec_value, b, RS_OPType.NCU_ER_P.dec_value, dst, 
                        RS_OPType.ALU_OUT.dec_value, ALUOUT_OPType.alu_none.dec_value,   
                        RS_OPType.ALU_OUT.dec_value, ALUOUT_OPType.alu_none.dec_value, RS_OPType.NCU_ER_P.dec_value, 0)
            else:
                raise SMT64ASMParseError(f"不支持的 CAL_OP 操作 {expr = }")

            self.field = CalField(op_code=self.op_type, fields=part)

            self.op_0 = alu_code
            self.op_1 = ALU_OPType.NOP_OP
            
            self.rs0 = a
            self.rs1 = b
            self.rd = dst
            
        elif optype == OPType.CALCU_IMM:
            """
            CalField(op_code=OPType.CALCU_IMM, fields=[21.3, 0,10, 0,6, 0,9, 0,7])
            """
            part = (a, 
                    rs_type.dec_value, b, RS_OPType.NCU_ER_P.dec_value, dst,   
                    RS_OPType.ALU_OUT.dec_value, ALUOUT_OPType.alu_none.dec_value, RS_OPType.NCU_ER_P.dec_value, 0)
            
            self.field = CalField(op_code=self.op_type, fields=part)

            self.op_0 = alu_code
            self.op_1 = ALU_OPType.NOP_OP
            
            self.rs0 = a
            self.rs1 = b
            self.rd = dst
             
    @classmethod
    def preprocess_expr(
        cls,
        a: list[Union[Register64, int, IEEE754]],
        op: str,
        b: list[Union[Register64, int, IEEE754]],
        expr: str,
    ) -> tuple[Union[int, IEEE754], OPType, ALU_OPType, RS_OPType, int]:
        """预处理表达式

        - 变换操作数顺序
        - 常数转换为 `IEEE754` 对象
        - 寄存器编号提取

        Example:
            >>> CAL_OP.preprocess_expr([3], "+", [Register64(4)], "R2 = 3 + R4")
            (<IEEE754:3:0.00>, <OPType.CALCU_IMM: 'CALCU_IMM'>, <ALU_OPType.ADD_OP: 'ADD_OP'>, <RS_OPType.NCU_ER_P: 'NCU_ER_P'>, <R4 = 0, used by: []>)
            >>> CAL_OP.preprocess_expr([IEEE754(3.0)], "+", [Register64(4)], "R2 = 3.0 + R4")
            (<IEEE754:1077936128:3.00>, <OPType.CALCU_IMM: 'CALCU_IMM'>, <ALU_OPType.ADD_OP: 'ADD_OP'>, <RS_OPType.NCU_ER_P: 'NCU_ER_P'>, <R4 = 0, used by: []>)

        Args:
            a (list[str, Union[Register64, int, IEEE754]]): 操作数 a
            op (str): 操作符
            b (list[str, Union[Register64, int, IEEE754]]): 操作数 b
            expr (str): 表达式, 用于报错

        Returns:
            tuple[Union[int, IEEE754], OPCode, int]: 常数或寄存器编号, 操作符, 寄存器编号
        """
        fp_type = ""
        fp_type += "-" if a[0] == "-" else "+"
        fp_type += "r" if isinstance(a[-1], Register64) else "i"
        fp_type += op
        fp_type += "-" if b[0] == "-" else "+"
        fp_type += "r" if isinstance(b[-1], Register64) else "i"

        if fp_type not in CAL_OP_CODES:
            raise SMT64ASMParseError(f"不支持的 CAL_OP 操作 {expr = }")

        action, optype, alu_code, rs_type = CAL_OP_CODES[fp_type]

        if action == "todo":
            raise NotImplementedError(f"暂不支持寄存器运算 {expr =}")

        if action == "swap":  # 交换 a, b 顺序实现相同结果
            a, b = b, a
        elif action == "swap-":  # 实现寄存器加负数, b[-1] 是负数的绝对值
            a, b = [-(b[-1])], a
        elif action == "swap+":  # 实现寄存器减负数, b[-1] 是负数的绝对值
            a, b = [b[-1]], a
        elif action == "":
            pass
        else:
            raise NotImplementedError(f"不支持的 CAL_OP 预处理操作 {action = }")
        # endregion: CAL_OP 预处理

        if isinstance(a[-1], (int, IEEE754)):  # 常数数值统一转换成 IEEE754 构造函数中处理
            a = IEEE754(a[-1] if a[0] != "-" else -a[-1])
        elif isinstance(a[-1], Register64):  # 寄存器编号
            a = a[-1]
        else:
            raise SMT64ASMParseError(f"不支持的 CAL_OP 操作 {expr = }, {a = }")

        # 经过预处理之后 b 只可能是寄存器数组
        b = b[-1]

        return a, optype, alu_code, rs_type, b

    @classmethod
    def parse_expr(cls, expr: str, regs: RegisterCollection) -> list[list[Union[str, int]]]:
        """解析表达式为 dst, a, op, b 型式.

        - 浮点数常数转换成 `IEEE754` 对象
        - 整数常数被看作为浮点数转换成 `IEEE754` 对象
        - 常数和寄存器的符号提取出来, 之后变换表达式的时候需要用到
        - R* 直接转换成寄存器对象

        Example:
            >>> regs = RegisterCollection()
            >>> CAL_OP.parse_expr("R2 = -3 + (-R3)", regs)
            [<R2 = 0, used by: []>, ['-', <IEEE754:1077936128:3.00>], '+', ['-', <R3 = 0, used by: []>]]
            >>> CAL_OP.parse_expr("R2 = -3 * R3", regs)
            [<R2 = 0, used by: []>, ['-', <IEEE754:1077936128:3.00>], '*', [<R3 = 0, used by: []>]]
            >>> CAL_OP.parse_expr("R2 = -3 + R3; R4 = 0 + R3", regs)
            [[<R2 = 0, used by: []>, ['-', <IEEE754:1077936128:3.00>], '+', [<R3 = 0, used by: []>]], [<R4 = 0, used by: []>, [<IEEE754:0:0.00>], '+', [<R3 = 0, used by: []>]]]

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

        dst = register
        a = (
            Group(Optional(sign) + reg_or_constant)  # 符号提取出来, 之后运算需要
            | Group(reg_or_constant)
            | nested_expr("(", ")", content=Optional(sign) + reg_or_constant)
        )
        b = a
        op = Literal("+") | Literal("-") | Literal("*") | Literal("/")
        assignment = dst + Suppress("=") + a + op + b

        result = []
        try:
            for one_expr in re.split(r";|\n", expr):
                if not (one_expr := one_expr.strip()):
                    continue
                ## NOTE, uger, 解析方式
                dst, a, op, b = assignment.parse_string(one_expr, parse_all=True).as_list()
                # 数值统一转换成 IEEE754 对象, 因为表达式中的常数不会是寄存器编号
                if isinstance(a[-1], (float, int)):
                    a[-1] = IEEE754(float(a[-1]))
                # NOTE, uger, fix the bug that constant is recognized as a register
                if isinstance(b[-1], (float, int)):
                    b[-1] = IEEE754(float(b[-1]))
                result.append([dst, a, op, b])
            return result
        except pp.ParseException as exc:
            raise SMT64ASMParseError(f"不支持的 CAL_OP 操作 {one_expr}") from exc

    @property
    def asm_value(self) -> str:
        """汇编语句

        Example:
            >>> regs = RegisterCollection()
            >>> program = '''
            ... R3 = 2 + R4
            ... R3 = -2 - R4
            ... R3 = R2 + R4
            ... R3 = R2 - R4
            ... R3 = 2 * R4
            ... R3 = -2 * R4
            ... R3 = R2 * R4
            ... R3 = -R2 * R4
            ... R3 = R4 * (-R2)
            ... '''
            >>> for op in CAL_OP.create_from_expr(program, regs):
            ...     op.asm_value
            'R3 = 2.00 + R4'
            'R3 = -2.00 - R4'
            'R3 = R2 + R4'
            'R3 = R2 - R4'
            'R3 = 2.00 * R4'
            'R3 = -2.00 * R4'
            'R3 = R2 * R4'
            'R3 = R4 * (-R2)'
            'R3 = R4 * (-R2)'
            >>> program_code = "; ".join(op.asm_value for op in CAL_OP.create_from_expr(program, regs))
            >>> program_code_code = "; ".join(op.asm_value for op in CAL_OP.create_from_expr(program_code, regs))
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
                (ALU_OPType.ADD_OP, RS_OPType.NCU_ER_P): "+", 
                (ALU_OPType.ADD_OP, RS_OPType.NCU_ER_N): "-", 
                (ALU_OPType.MUL_OP, RS_OPType.NCU_ER_P): "*", 
                (ALU_OPType.MUL_OP, RS_OPType.NCU_ER_N): "* (-)", 
                (ALU_OPType.DIV_OP, RS_OPType.NCU_ER_P): "/", 
                (ALU_OPType.DIV_OP, RS_OPType.NCU_ER_N): "/ (-)", 
            }[(self.op_0, self.rs_type)]
            
            asm.append(
                {
                    OPType.CALCU_IMM: f"{dst} = {a} {sign} {b}",
                    OPType.CALCU_REG: f"{dst} = {a} {sign} {b}",
                }[self.op_type]
            )

        if not asm:
            raise ValueError("没有 CAL_OP 操作")

        return ", ".join(asm)


__all__ = ["CAL_OP"]

