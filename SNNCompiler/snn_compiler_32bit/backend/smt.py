# pylint: disable=too-few-public-methods, protected-access, invalid-name

"""SMT 指令模型
"""
from dataclasses import dataclass
from typing import List, Tuple, Union

from ..common.operator import Operator
from ..common.register import Register


@dataclass
class SMT:  # pylint: disable=too-many-instance-attributes
    """SMT 指令模型."""

    op: Operator = Operator.NOP
    """6-bit 操作码.
    """

    a1: Register = None
    """5-bit 加法源操作数 A1.
    """

    a2: Register = None
    """5-bit 加法源操作数 A2.
    """

    m1: Register = None
    """5-bit 乘法源操作数 M1.
    """

    m2: Register = None
    """5-bit 乘法源操作数 M2.
    """

    s: Register = None
    """3-bit 加法目标操作数 S. Default to register with index 7 means not applicable.
    """

    p: Register = None
    """3-bit 乘法目标操作数 P. Default to register with index 7 means not applicable.
    """

    operator: str = "unknown"
    """记录加法 `add`,  乘法 `mul` 或者乘加 `add_mul`.
    """

    result_bits: int = 3
    """结果寄存器位宽. 3 或者 4.
    """

    @property
    def op_bits(self) -> int:
        """操作数位宽. 6 或者 4."""
        if self.result_bits == 3:
            return 6
        if self.result_bits == 4:
            return 4
        raise NotImplementedError(f"暂不支持 {self.result_bits} 位结果.")

    input_bits: int = 5
    """输入寄存器位宽. 5.
    """

    @property
    def input_regs(self) -> List[Register]:
        """返回输入寄存器. `self.a1`, `self.a2`, `self.m1`, `self.m2`.

        Returns:
            list[Register]: 所有输入寄存器.
        """
        return [self.a1, self.a2, self.m1, self.m2]

    def update_regs(
        self,
        old_reg: Register,
        new_reg: Register,
        reg_names: List[str] = None,
    ) -> bool:
        """更新寄存器.

        Args:
            old_reg (Register): 旧寄存器.
            new_reg (Register): 新寄存器.

        Returns:
            bool: 更新寄存器成功.
        """
        reg_names = reg_names or ["a1", "a2", "m1", "m2", "s", "p"]
        result = False
        for reg_name in reg_names:
            if old_reg != getattr(self, reg_name, None):
                continue
            setattr(self, reg_name, new_reg)
            result = True
        return result

    def update_operand(self, old_reg: Register, new_reg: Register) -> bool:
        """更新操作数寄存器.

        Args:
            old_reg (Register): 旧的寄存器.
            new_reg (Register): 新的寄存器.

        Returns:
            bool: 更新了操作数.
        """
        if self.op != Operator.CALCU:
            return False
        return self.update_regs(old_reg=old_reg, new_reg=new_reg, reg_names=["a1", "a2", "m1", "m2"])

    @property
    def value(self) -> int:
        """只读 SMT 值. 根据当前 SMT 计算.

        Returns:
            int: 当前 SMT 的值.
        """
        result = []
        result += [bin(self.op.value)[2:].rjust(self.op_bits, "0")]

        for reg in ["a1", "a2", "s", "m1", "m2", "p"]:
            width = self.input_bits
            if reg in ["s", "p"]:
                width = self.result_bits
            if (reg_value := getattr(self, reg, None)) is None:
                result += ["0".rjust(width, "0")]
            else:
                actual_width = len(bin(reg_value.index)[2:])
                if width < actual_width:
                    raise RuntimeError(f"寄存器 {reg} {reg_value.index = } 宽度 {actual_width} > {width}")
                result += [bin(reg_value.index)[2:].rjust(width, "0")]

        return "_".join(result)

    @property
    def reg_result(self) -> Union[Register, Tuple[Register, Register]]:
        """结果寄存器.

        Returns:
            Union[Register, Tuple[Register, Register]]: 结果寄存器.
                如果操作为加乘则返回加法结果和乘法结果两个寄存器.
        """
        if self.op != Operator.CALCU:
            raise ValueError(f"操作 {self} 没有结果寄存器")

        if self.operator == "add":
            return self.s

        if self.operator == "mul":
            return self.p

        if self.operator == "add_mul":
            return self.s, self.p

        raise ValueError(f"不能得到结果寄存器: 未知操作 {self.operator}")

    @reg_result.setter
    def reg_result(self, value: Register) -> Register:
        """设置结果寄存器.

        Raises:
            NotImplementedError: 暂不支持乘法和加法同时运算.

        Returns:
            Register: 结果寄存器
        """
        if self.op != Operator.CALCU:
            raise ValueError(f"{self.op} 没有结果寄存器")

        if self.operator == "add":
            self.s = value
            return

        if self.operator == "mul":
            self.p = value
            return

        if self.operator == "add_mul":
            if isinstance(value, tuple) and len(value) == 2:
                self.s = value[0]
                self.p = value[1]
                return
            raise ValueError(f"运算的结果不能为 {value}.")

        raise ValueError(f"不能设置结果寄存器: 未知操作 {self.operator}")

    def __str__(self) -> str:
        if self.op == Operator.CALCU:
            sum_product = []
            if (self.a1.alias, self.a2.alias) != ("ZERO_REG", "ZERO_REG"):
                sum_product += [f"{self.s.short} = {self.a1.short} + {self.a2.short}"]
            if (self.m1.alias, self.m2.alias) != ("ZERO_REG", "ZERO_REG"):
                sum_product += [f"{self.p.short} = {self.m1.short} * {self.m2.short}"]
            return f"{self.op.name}: " + (", ".join(sum_product))
        if self.op == Operator.V_SET:
            return f"V_SET: delta V = {self.a1.name}, V_reset = {self.a2.name}:{self.a2.value}, V = {self.s.name}"
        if self.op == Operator.SPIKE:
            return f"SPIKE: delta V = {self.a1.name}"
        if self.op in [Operator.NOP, Operator.SRAM_LOAD, Operator.SRAM_SAVE, Operator.END, Operator.SPIKE]:
            return self.op.name
        result = [str(self.op)]
        for key in ["a1", "a2", "s", "m1", "m2", "p"]:
            reg = getattr(self, key)
            result += [f"{key.upper()}:{reg.name}({reg.index})"]
        return ", ".join(result)

    def __repr__(self) -> str:
        return self.__str__()
