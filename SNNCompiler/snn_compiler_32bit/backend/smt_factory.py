# pylint: disable=too-few-public-methods, protected-access, invalid-name

"""SMT 指令模型
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union

from ..common.operator import Operator
from ..common.register import Register
from ..common.register_collection import RegisterCollection
from .smt import SMT


@dataclass
class SMTFactory:
    """SMT 生成器."""

    regs: RegisterCollection
    """编译用到的寄存器
    """

    def get_reg(self, int_or_reg: Union[Register, int]) -> Register:
        """根据数值或寄存器返回寄存器对象.

        Args:
            int_or_reg (Register | int): 寄存器对象或寄存器数值. 支持 0 和 1.

        Returns:
            Register: 寄存器对象.
        """
        if isinstance(int_or_reg, Register):
            return int_or_reg

        if int_or_reg == 0:
            return self.regs.ZERO_REG

        if int_or_reg == 1:
            return self.regs.ONE_REG

        raise ValueError(f"不支持的 {int_or_reg = }")

    def add(self, a: Union[Register, int], b: Union[Register, int], c: Optional[Register] = None) -> List[SMT]:
        """加法 c = a + b. MUL_P = 0 * 0.

        Args:
            a (Union[Register, int]): 被加数寄存器, 0: SMTReg.ZERO_REG, 1: SMTReg.ONE_REG
            b (Union[Register, int]): 加数寄存器, 0: SMTReg.ZERO_REG, 1: SMTReg.ONE_REG
            c (Register, optional): 结果寄存器.

        Returns:
            list[SMT]: SMT 语句
        """

        result = [
            SMT(
                op=Operator.CALCU,
                a1=self.get_reg(a),
                a2=self.get_reg(b),
                s=c or self.regs.ADD_S,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG,
                operator="add",
            ),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),  ## uger
        ]
        return result

    def move(self, src: Register, dst: Register) -> List[SMT]:
        """通过加零来移动 `src` 寄存器的值到 `dst` 寄存器.

        Args:
            src (Register): 源寄存器.
            dst (Register): 目标寄存器.

        Returns:
            list[SMT]: SMT 语句
        """
        return self.add(src, 0, dst)

    def multiply(self, a: Union[Register, int], b: Union[Register, int], c: Optional[Register] = None) -> List[SMT]:
        """乘法 ADD_S = 0 + 0, c = a * b,

        Args:
            a (Register | int): 被乘数数寄存器, 0: SMTReg.ZERO_REG, 1: SMTReg.ONE_REG
            b (Register | int): 乘数寄存器, 0: SMTReg.ZERO_REG, 1: SMTReg.ONE_REG
            c (Register, optional): 结果寄存器

        Returns:
            list[SMT]: SMT 语句
        """

        result = [
            SMT(
                op=Operator.CALCU,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.get_reg(a),
                m2=self.get_reg(b),
                p=c or self.regs.MUL_P,
                operator="mul",
            ),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),  ## uger
        ]
        return result

    # pylint: disable-next=too-many-arguments
    def add_multiply(
        self,
        a1: Register,
        a2: Register,
        m1: Register,
        m2: Register,
        s: Optional[Register] = None,
        p: Optional[Register] = None,
    ) -> List[SMT]:
        """加法和乘法同时运算.

        - `s = a1 + a2`
        - `p = m1 * m2`

        Args:
            a1 (Register | int): 被加数数寄存器, 0: SMTReg.ZERO_REG, 1: SMTReg.ONE_REG
            a2 (Register | int): 加数寄存器, 0: SMTReg.ZERO_REG, 1: SMTReg.ONE_REG
            s (Register, optional): 和结果寄存器
            m1 (Register | int): 被乘数数寄存器, 0: SMTReg.ZERO_REG, 1: SMTReg.ONE_REG
            m2 (Register | int): 乘数寄存器, 0: SMTReg.ZERO_REG, 1: SMTReg.ONE_REG
            p (Register, optional): 积结果寄存器

        Returns:
            list[SMT]: SMT 语句
        """

        result = [
            SMT(
                op=Operator.CALCU,
                a1=a1,
                a2=a2,
                s=s or self.regs.ADD_S,
                m1=m1,
                m2=m2,
                p=p or self.regs.MUL_P,
                operator="add_mul",
            ),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),  ## uger
        ]
        return result

    def sram_load(self) -> List[SMT]:
        """读取 SRAM.

        Returns:
            list[SMT]: SMT 语句
        """
        return [SMT(
                op=Operator.SRAM_LOAD,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),
            SMT(op=Operator.NOP,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG),  ## uger
        ]

    def v_set(self, delta_v: Register, v_reset: Register) -> List[SMT]:
        """更新 V

        Args:
            delta_v (Register): V_thresh - V 结果寄存器
            v_reset (Register): v_reset 寄存器

        Returns:
            list[SMT]: SMT 语句
        """
        return [SMT(op=Operator.V_SET, a1=delta_v, a2=v_reset, s=self.regs.V,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG)
            ]
    def spike(self, delta_v: Register) -> List[SMT]:
        """Spike

        Args:
            delta_v (Register): V_thresh - V 结果寄存器

        Returns:
            list[SMT]: SMT 语句
        """
        return [SMT(Operator.SPIKE, a1=delta_v,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG)
            ]

    def sram_save(self) -> List[SMT]:
        """存储 SRAM.

        Returns:
            list[SMT]: SMT 语句
        """
        return [SMT(Operator.SRAM_SAVE,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG)
            ]

    def end(self) -> List[SMT]:
        """结束.

        Returns:
            list[SMT]: SMT 语句
        """
        return [SMT(Operator.END,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.NONE_REG,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.NONE_REG)
            ]