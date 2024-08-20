"""SMT 64位 NOP指令"""
# 2024.07@uger
from __future__ import annotations

from ...common.smt_64_reg import RegisterCollection
from ..smt_64_op import OPType, CalField, ALU_OPType
from .smt64 import SMT64, SMT64ASMParseError


class NOP(SMT64):
    """NOP 指令

    Example:
        >>> regs = RegisterCollection()
        >>> NOP()
        NOP
        >>> list(NOP.create_from_expr("NOP; NOP; NOP", regs))
        [NOP, NOP, NOP]
    """

    # pylint: disable-next=unused-argument
    def __init__(self, expr: str = None, regs: RegisterCollection = None) -> None:
        """构造函数

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合
        """
        super().__init__(op_type=OPType.NOP)

        if expr and (expr.upper() != "NOP"):
            raise SMT64ASMParseError(f"不支持的操作 {expr}")

        field_value = CalField(op_code = OPType.NOP)

        super().__init__(
            op_type = OPType.NOP,
            op_0 = ALU_OPType.ENABLE_OP,
            op_1 = ALU_OPType.ENABLE_OP,
            field = field_value, 
        )

    @classmethod
    def parse_expr(cls, expr: str, regs: RegisterCollection) -> list[str]:
        """解析表达式

        Example:
            >>> regs = RegisterCollection()
            >>> NOP.parse_expr("NOP", regs)
            ['NOP']
            >>> list(NOP.create_from_expr("NOP; NOP; NOP", regs))[0].asm_value
            'NOP'
            >>> NOP.parse_expr("NOP, R=3", regs)
            Traceback (most recent call last):
            ...
            snn_compiler.backend.smt_64_stmt.smt64.SMT64ASMParseError: 不支持的操作 R=3

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合

        Returns:
            list[str]: 解析后的指令列表, e.g. `["nop", "nop"]`
        """
        result: list[str] = []
        for nop in expr.split(","):
            if (nop := nop.strip().upper()) == "NOP":
                result.append(nop)
                continue
            raise SMT64ASMParseError(f"不支持的操作 {nop}")
        return result

    @property
    def asm_value(self) -> str:
        """汇编语句

        Returns:
            str: 汇编语句
        """
        return "NOP"


__all__ = ["NOP"]
