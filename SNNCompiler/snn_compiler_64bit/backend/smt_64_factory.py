"""这个模块包含了用于处理 SMT 64 位指令的常用函数和类
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..common.smt_64_reg import Register64, RegisterCollection, RegOrConstant
from .smt_64_stmt import CAL_OP, EXPLOG_OP, SMT64


@dataclass
class SMT64Factory:
    """64-bit SMT 语句生成器."""

    regs: RegisterCollection = field(default_factory=RegisterCollection)
    """寄存器集合"""

    def get_expr(self, a: RegOrConstant, b: RegOrConstant, c: Register64, op: str) -> list[SMT64]:
        """获取表达式

        - 寄存器对象不变
        - 常数转换为 IEEE754 格式

        >>> regs = RegisterCollection()
        >>> SMT64Factory(regs).get_expr(0, regs[3], regs[0], "+")
        'R0 = 0 + R3'
        >>> SMT64Factory(regs).get_expr(0, regs[4], regs[1], "+")
        'R1 = 0 + R4'
        >>> SMT64Factory(regs).get_expr(regs[6], regs[3], regs[2], "+")
        'R2 = R6 + R3'

        Args:
            a RegOrConstant: 第一个操作数
            b RegOrConstant: 第二个操作数
            c (Register, optional): 结果寄存器
            op (str): 操作符

        Returns:
            str: 表达式
        """
        if not isinstance(c, Register64):
            raise ValueError(f"表达式不支持类型 {c = }, 只支持 Register64")

        a = SMT64.get_asm_value(a)
        b = SMT64.get_asm_value(b)
        return f"R{c.index} = {a} {op} {b}"

    def get_explog_expr(self, a: RegOrConstant, c: Register64, op: str) -> list[SMT64]:
        """获取表达式

        - 寄存器对象不变
        - 常数转换为 IEEE754 格式

        >>> regs = RegisterCollection()
        >>> SMT64Factory(regs).get_explog_expr(regs[0], regs[3], 'exp')
        'R0 = exp(R3)'
        >>> SMT64Factory(regs).get_explog_expr(regs[4], regs[1], 'log')
        'R1 = log(R4)'

        Args:
            a RegOrConstant: 指数对数操作数
            c (Register, optional): 结果寄存器

        Returns:
            str: 表达式
        """
        if not isinstance(c, Register64):
            raise ValueError(f"表达式不支持类型 {c = }, 只支持 Register64")

        a = SMT64.get_asm_value(a)
        return f"R{c.index} = {op}({a})"


    def add(self, a: RegOrConstant, b: RegOrConstant, c: Register64) -> list[SMT64]:
        """加法 c = a + b. 

        >>> regs = RegisterCollection()
        >>> SMT64Factory(regs).add(0, regs[3], regs[0])
        [R0 = 0.00 + R3]
        >>> SMT64Factory(regs).add(0, regs[4], regs[1])
        [R1 = 0.00 + R4]
        >>> SMT64Factory(regs).add(regs[6], regs[3], regs[2])
        [R2 = R6 + R3]

        Args:
            a RegOrConstant: 被加数寄存器或者数值
            b RegOrConstant: 加数寄存器
            c (Register, optional): 结果寄存器

        Returns:
            list[SMT]: SMT 语句
        """
        return [CAL_OP(self.get_expr(a, b, c, "+"), self.regs)]

    def subtract(self, a: RegOrConstant, b: RegOrConstant, c: Register64) -> list[SMT64]:
        """减法 c = a - b. 

        >>> regs = RegisterCollection()
        >>> SMT64Factory(regs).subtract(0, regs[3], regs[0])
        [R0 = 0.00 - R3]
        >>> SMT64Factory(regs).subtract(0, regs[4], regs[1])
        [R1 = 0.00 - R4]
        >>> SMT64Factory(regs).subtract(regs[6], regs[3], regs[2])
        [R2 = R6 - R3]

        Args:
            a RegOrConstant: 被减数寄存器或者数值
            b RegOrConstant: 减数寄存器
            c (Register, optional): 结果寄存器

        Returns:
            list[SMT64]: SMT 语句
        """
        return [CAL_OP(self.get_expr(a, b, c, "-"), self.regs)]

    def multiply(self, a: RegOrConstant, b: RegOrConstant, c: Register64) -> list[SMT64]:
        """乘法 c = a * b. 

        >>> regs = RegisterCollection()
        >>> SMT64Factory(regs).multiply(0, regs[3], regs[0])
        [R0 = 0.00 * R3]
        >>> SMT64Factory(regs).multiply(0, regs[4], regs[1])
        [R1 = 0.00 * R4]
        >>> SMT64Factory(regs).multiply(regs[6], regs[3], regs[2])
        [R2 = R6 * R3]

        Args:
            a RegOrConstant: 被乘数寄存器或者数值
            b RegOrConstant: 乘数寄存器
            c (Register, optional): 结果寄存器

        Returns:
            list[SMT64]: SMT 语句
        """
        return [CAL_OP(self.get_expr(a, b, c, "*"), self.regs)]

    def divide(self, a: RegOrConstant, b: RegOrConstant, c: Register64) -> list[SMT64]:
        """除法 c = a / b. 

        >>> regs = RegisterCollection()
        >>> SMT64Factory(regs).divide(15.0, regs[3], regs[0])
        [R0 = 15.0 / R3]
        >>> SMT64Factory(regs).divide(regs[3], regs[4], regs[1])
        [R1 = R3 / R4]

        Args:
            a RegOrConstant: 被除数存器或者数值
            b RegOrConstant: 除数寄存器
            c (Register, optional): 结果寄存器

        Returns:
            list[SMT64]: SMT 语句
        """
        return [CAL_OP(self.get_expr(a, b, c, "/"), self.regs)]

    def exponential(self, a: RegOrConstant, c: Register64) -> list[SMT64]:
        """指数运算 c = exp(a).

        >>> regs = RegisterCollection()
        >>> SMT64Factory(regs).exponential(regs[3], regs[0])
        [R0 = exp(R3)]
        >>> SMT64Factory(regs).exponential(100.0, regs[1])
        [R1 = exp(100.0)]

        Args:
            a RegOrConstant: 寄存器或者立即数
            c (Register, optional): 结果寄存器

        Returns:
            list[SMT64]: SMT 语句
        """
        return [EXPLOG_OP(self.get_explog_expr(a, c, "exp"), self.regs)]
    
    def log(self, a: RegOrConstant, c: Register64) -> list[SMT64]:
        """对数运算 c = ln(a).

        >>> regs = RegisterCollection()
        >>> SMT64Factory(regs).exponential(regs[3], regs[0])
        [R0 = log(R3)]
        >>> SMT64Factory(regs).exponential(100.0, regs[1])
        [R1 = log(100.0)]

        Args:
            a RegOrConstant: 寄存器或者立即数
            c (Register, optional): 结果寄存器

        Returns:
            list[SMT64]: SMT 语句
        """
        return [EXPLOG_OP(self.get_explog_expr(a, c, "log"), self.regs)]


    # def move(self, src: Register64, dst: Register64) -> list[SMT64]:
    #     """移动寄存器 src -> dst

    #     >>> regs = RegisterCollection()
    #     >>> SMT64Factory(regs).move(regs[1], regs[0])
    #     [R1 -> R0]
    #     >>> SMT64Factory(regs).move(regs[3], regs[2])
    #     [R3 -> R2]

    #     Args:
    #         src (Register): 源寄存器
    #         dst (Register): 目标寄存器

    #     Returns:
    #         list[SMT64]: SMT 语句
    #     """
    #     return [REG_OP(f"R{dst.index} = R{src.index}", self.regs)]

__all__ = ["SMT64Factory"]
