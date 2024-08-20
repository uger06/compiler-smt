"""寄存器集合
"""
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Set, Tuple, Union

from addict import Dict as AttrDict

from .register import Register


@dataclass
class RegisterCollectionBase:  # pylint: disable=too-many-instance-attributes
    """寄存器集合基类"""

    # region: 预设寄存器成员
    R0: Register = None
    """本地寄存器. 不可以用作输入, 可以用作结果寄存器.
    """

    R1: Register = None
    """本地寄存器. 不可以用作输入, 可以用作结果寄存器.
    """

    R2: Register = None
    """本地寄存器. 可以用作输入, 可以用作结果寄存器.
    """

    R3: Register = None
    """本地寄存器. 可以用作输入, 可以用作结果寄存器.
    """

    R4: Register = None
    """本地寄存器. 可以用作输入, 可以用作结果寄存器.
    """

    R5: Register = None
    """本地寄存器. 可以用作输入, 可以用作结果寄存器.
    """

    R6: Register = None
    """本地寄存器. 可以用作输入, 可以用作结果寄存器.
    """

    R7: Register = None
    """本地寄存器. 可以用作输入, 不可以用作结果寄存器.
    """

    R8: Register = None
    """本地寄存器. 可以用作输入, 不可以用作结果寄存器.
    """

    R9: Register = None
    """本地寄存器. 可以用作输入, 不可以用作结果寄存器.
    """

    R5_NEG: Register = None
    """本地寄存器. 存储 R5 的负值. 不可以用作输入, 不可以用作结果寄存器.
    """

    R6_NEG: Register = None
    """本地寄存器. 存储 R6 的负值. 不可以用作输入, 不可以用作结果寄存器.
    """

    ADD_S: Register = None
    """加法结果本地寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    MUL_P: Register = None
    """乘加法结果本地寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR0: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR1: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR2: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR3: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR4: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR5: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR6: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR7: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR8: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR9: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR10: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR11: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR12: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR13: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR14: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR15: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR16: Register = None
    """共享寄存器. 存储常数 1. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR17: Register = None
    """共享寄存器. 存储常数 0. 不可以用作输入, 不可以用作结果寄存器.
    """
    # endregion: 预设寄存器成员

    all_registers: Dict[int, Register] = None
    """所有寄存器.
    """

    result_bits: int = 3
    """结果寄存器位数, 3 或者 4
    """

    valid_result_regs: List[Register] = None
    """可以使用的结果寄存器:

    - `result_bits == 3`: R0, R1, R2, R3, R4
        - R7 = 只输出到 ADD_S 和 MUL_P
    - `result_bits == 4`: R0, R1, R2, R3, R4, R7, R8, R9
        - R15 = 只输出到 ADD_S 和 MUL_P
    """

    valid_func_arg_regs: List[Register] = None
    """可以使用的函数输入寄存器:

    - R2, R3, R4, R7, R8, R9
    - V 输入固定为 R5 或者 R6. 不算在函数输入寄存器之内.
    """

    def get_reg_by_name(self, name: Union[str, List[str]]) -> Union[Register, List[Register]]:
        """根据寄存器名字找到寄存器. 必须在寄存器初始化之后执行.

        Args:
            name (str | list[str]): 寄存器名字或名字列表.

        Returns:
            Register | list[Register]: 找到的寄存器或者寄存器列表.
        """

        return self._get_reg(text=name, text_type="name")

    def get_reg_by_alias(self, alias: Union[str, List[str]]) -> Union[Register, List[Register]]:
        """根据寄存器名字找到寄存器. 必须在寄存器初始化之后执行.

        Args:
            alias (str | list[str]): 寄存器名字或名字列表.

        Returns:
            Register | list[Register]: 找到的寄存器或者寄存器列表.
        """

        return self._get_reg(text=alias, text_type="alias")

    def _get_reg(self, text: Union[str, List[str]], text_type: str = "name") -> Union[Register, List[Register]]:
        """根据寄存器名字找到寄存器. 必须在寄存器初始化之后执行.

        Args:
            text (str | list[str]): 查找的文本或文本列表.
            text_type (str): 查找类型, 支持 `"name"` 或者 `"alias"`.

        Returns:
            Register | list[Register]: 找到的寄存器或者寄存器列表.
        """

        if not isinstance(text, (str, list)):
            raise ValueError(f"{text = } 不是 str 或者 list.")

        if self.all_registers is None:
            raise RuntimeError("寄存器没有初始化. 先运行 `self.reset()`.")

        if isinstance(text, str):
            for reg in self.all_registers.values():
                if text_type == "name":
                    if reg.name == text:
                        return reg
                elif text_type == "alias":
                    if reg.alias == text:
                        return reg
                else:
                    raise NotImplementedError(f"不支持 {text_type = }")
            raise ValueError(f"找不到寄存器 {text =}, {text_type = }")

        result = sorted([self._get_reg(t, text_type) for t in text], key=lambda r: r.index)
        return result

    @property
    def default_valid_result_regs(self) -> List[Register]:
        """默认的可使用结果寄存器.

        `self.result_bits`:

        - 3-bit: R0, R1, R2, R3, R4
        - 4-bit: R0, R1, R2, R3, R4, R8, R9

        Returns:
            list[Register]: 默认的可使用结果寄存器.
        """
        if self.result_bits == 3:
            reg_names = ["R0", "R1", "R2", "R3", "R4"]
        elif self.result_bits == 4:
            reg_names = ["R0", "R1", "R2", "R3", "R4", "R8", "R9"]
        else:
            raise NotImplementedError(f"暂不支持 {self.result_bits} 位结果.")

        return self.get_reg_by_name(list(reversed(reg_names)))

    @property
    def default_valid_func_arg_regs(self) -> List[Register]:
        """默认的可使用函数输入寄存器: R2, R3, R4, (R7), R8, R9.

        `result_bits == 4` 时可以使用 R7.

        Returns:
            list[Register]: 默认的可使用函数输入寄存器.
        """
        reg_names = ["R2", "R3", "R4", "R8", "R9"]
        if self.result_bits == 3:
            pass
        elif self.result_bits == 4:
            reg_names += ["R7"]
        else:
            raise NotImplementedError(f"暂不支持 {self.result_bits} 位结果.")

        reg_names = sorted(reg_names)

        result = sorted(self.get_reg_by_name(reg_names), key=lambda r: r.name)
        return result

    def reset(self) -> None:
        """初始化所有寄存器"""

        # region: all_registers
        self.all_registers = AttrDict()

        local_regs = [f"R{i}" for i in range(10)]
        special_local_regs = ["R5_NEG", "R6_NEG", "ADD_S", "MUL_P"]
        shared_regs = [f"SR{i}" for i in range(18)]

        index = 0
        for name in local_regs + special_local_regs + shared_regs:
            if getattr(self, name, None) is None:
                setattr(self, name, Register().update(index=index, name=name))
            self.all_registers[index] = getattr(self, name)
            index += 1
        # endregion: all_registers

        self.valid_result_regs = self.default_valid_result_regs
        self.valid_func_arg_regs = self.default_valid_func_arg_regs

    def __post_init__(self) -> None:
        """`dataclass` 构造函数, `__init__` 运行之后执行"""
        self.reset()


class RegisterCollection(RegisterCollectionBase):
    """寄存器集合, 用于 SMT 编译器.

    - V 可以是 R5 或者 R6.
    - NONE_REG 作为运算结果使用的话只存储在 ADD_S 和 MUL_P.
    - ONE_REG 储存常数 1
    - ZERO_REG 储存常数 0
    - pos_reg 为取负值的输入
    - neg_reg 为取负值的输出
    """

    def use_reg(self, reg: Register, used_by: int = None) -> Register:
        """使用寄存器 `reg`.
        如果寄存器在 `self.valid_result_regs` 或者 `self.valid_func_arg_regs` 则从其中删除.

        Args:
            reg (Register): 要使用的寄存器对象.
            used_by (int, optional): 使用的语句序列号, 默认为 `None`.

        Returns:
            Register: 使用的寄存器.
        """
        if reg in self.valid_result_regs:
            self.valid_result_regs.remove(reg)
        if reg in self.valid_func_arg_regs:
            self.valid_func_arg_regs.remove(reg)
        if used_by is not None:
            reg.used_by.add(used_by)
        return reg

    def release_reg(self, reg: Register) -> Register:
        """不使用寄存器 `reg`.

        Args:
            reg (Register): 要使用的寄存器对象.
            used_by (int, optional): 使用的语句序列号, 默认为 `None`.

        Returns:
            Register: 释放的寄存器.
        """
        reg.release()
        if reg in self.default_valid_result_regs:
            self.valid_result_regs.append(reg)
        if reg in self.default_valid_func_arg_regs:
            self.valid_func_arg_regs.append(reg)
        return reg

    # region: V 和 V_NEG
    _v: Register = None
    """函数参数寄存器 V.
    """

    _v_neg: Register = None
    """函数参数寄存器 V 的负值.
    """

    @property
    def V(self) -> Register:
        """函数参数寄存器 V.

        Returns:
            Register: 函数参数寄存器 V.
        """
        if self._v is None:
            # self.V = self.R5
            # TODO, uger, 固定R6作为V
            self.V = self.R6
        return self._v

    @V.setter
    def V(self, value: Register) -> None:
        """设置函数参数寄存器 V.

        Args:
            value (Register): 函数参数寄存器 V.
        """
        if value not in [self.R5, self.R6]:
            raise RuntimeError(f"V 只能是 R5 或者 R6 而不是 {value}")

        if value == self.R6:
            self._v = self.R6.update(alias="V", used_by={-1})
            self._v_neg = self.R6_NEG.update(alias="V_NEG", used_by={-1})
            self.release_reg(self.R5)
            self.release_reg(self.R5_NEG)
        else:
            self._v = self.R5.update(alias="V", used_by={-1})
            self._v_neg = self.R5_NEG.update(alias="V_NEG", used_by={-1})
            self.release_reg(self.R6)
            self.release_reg(self.R6_NEG)

        self.use_reg(self._v)
        self.use_reg(self._v_neg)

    @property
    def V_NEG(self) -> Register:
        """函数参数寄存器 V 的负数.

        Returns:
            Register: 函数参数寄存器 V 的负数.
        """
        if self._v_neg is None:
            # self.V = self.R5
            # TODO, uger, 固定R6作为V
            self.V = self.R6
        return self._v_neg

    @V_NEG.setter
    def V_NEG(self, value: Register) -> None:
        """设置函数参数寄存器 V 的负数.

        Args:
            value (Register): 函数参数寄存器 V 的负数.
        """
        if value not in [self.R5_NEG, self.R6_NEG]:
            raise RuntimeError(f"V 只能是 R5 或者 R6 而不是 {value}")

        if value == self.R6_NEG:
            self.V = self.R6
        else:
            self.V = self.R5

    # endregion: V 和 V_NEG

    @cached_property
    def NONE_REG(self) -> Register:
        """占位符寄存器. R7 或者 R15. 不可以用作输入, 可以用作结果寄存器."""
        if self.result_bits == 3:
            return Register(index=7, _name="NONE_REG")
        if self.result_bits == 4:
            return Register(index=15, _name="NONE_REG")
        raise NotImplementedError(f"暂不支持 {self.result_bits} 位结果.")

    @cached_property
    def ONE_REG(self) -> Register:
        """常数 1, 使用未占用的 SR."""
        result = self.unused_shared_reg.update(alias="ONE_REG", value=1, used_by={-2})
        self.use_reg(result)
        return result

    @cached_property
    def ZERO_REG(self) -> Register:
        """常数 0, 使用未占用的 SR."""
        result = self.SR17.update(alias="ZERO_REG", value=0, used_by={-2})
        self.use_reg(result)
        return result

    @property
    def pos_reg(self) -> Tuple[Register, Register]:
        """取负数使用的正值寄存器.

        Returns:
            Register: 正值寄存器.
        """
        if self.V == self.R5:
            return self.R6
        return self.R5

    @property
    def neg_reg(self) -> Tuple[Register, Register]:
        """取负数使用的负值寄存器.

        Returns:
            Register: 负值寄存器.
        """
        if self.V == self.R5:
            return self.R6_NEG
        return self.R5_NEG

    @property
    def used_shared_reg(self) -> Set[Register]:
        """返回被使用的共享寄存器 (`SR0` 到 `SR17`).

        Returns:
            set[Register]: 被使用的共享寄存器.
        """
        result = set()
        for sr in self.all_registers.values():
            if not sr.name.startswith("SR"):
                continue
            if sr.used_by:
                result.add(sr)
        return result

    @property
    def unused_shared_reg(self) -> Register:
        """返回一个还没有被使用的共享寄存器 (`SR0` 到 `SR17`).

        Returns:
            Register: 还没有被使用的共享寄存器.
        """
        for sr in self.all_registers.values():
            if not sr.name.startswith("SR"):
                continue
            if not sr.used_by:
                return self.use_reg(sr)
        raise RuntimeError("找不到没有使用过的共享寄存器.")

    @property
    def unused_dummy_reg(self) -> Register:
        """返回一个虚拟寄存器 (`DUMMY_*`).
        最后这些虚拟寄存器会被合并到 R0-6.

        Returns:
            Register: 还没有被使用的虚拟寄存器.
        """
        index = len(self.all_registers)
        result = Register(_name=f"DUMMY_{index}", index=index)
        self.all_registers[index] = result
        return result

    @property
    def unused_arg_reg(self) -> Register:
        """返回一个还没有被使用的函数参数寄存器 (R2-4, R7-9).

        Returns:
            Register: 还没有被使用的函数参数寄存器.
        """
        for reg in reversed(self.valid_func_arg_regs):
            if reg.used_by:
                continue
            self.use_reg(reg)
            return reg
        raise RuntimeError("找不到未被占用的函数参数寄存器")

    @property
    def dummy_regs(self) -> List[Register]:
        """返回所有被使用的虚拟寄存器.

        Returns:
            list[Register]: 所有被使用的虚拟寄存器.
                用第一次使用的 IR 语句排序.
        """
        result = []

        for reg in self.all_registers.values():
            if not reg.name.startswith("DUMMY_"):
                continue
            if not reg.used_by:  # 跳过没用的结果寄存器
                continue
            result += [reg]
        result = sorted(result, key=lambda r: r.first)
        return result
