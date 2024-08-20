"""寄存器模型
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RegisterBase:
    """32 位寄存器基类"""

    index: int = -1
    """Register index. Defaults to -1 means last Register index + 1.
    """

    used_by: set[int] = field(default_factory=set)
    """使用此寄存器的 IR 语句:

    - int: 语句的索引值, -1 表示函数参数, -2 表示保留寄存器 0 和 1: `ZERO_REG` 和 `ONE_REG`
    """

    alias: str = ""
    """SMT 语句中的寄存器别名, e.g. `V_reset`.
    """

    as_arg: str = ""
    """函数输入的名字, e.g. `I`."""

    as_return: str = ""
    """函数输出的名字, e.g. `I`."""

    _name: str = ""
    """汇编语言中的寄存器名称, e.g. R0. 只读属性."""

    @property
    def name(self) -> str:
        """汇编语言中的寄存器名称, e.g. R0."""
        return self._name

    def update_name(self, value: str) -> None:
        """更新 `self._name`.

        Args:
            value (str): 名字.
        """
        self._name = value

    @property
    def short(self) -> str:
        """寄存器的别名或者名称

        Returns:
            str: 寄存器的别名或者名称
        """
        if self.as_arg:
            return self.as_arg
        if self.alias:
            return self.alias
        return self.name

    value: int = 0
    """当前寄存器的值. Defaults to 0.
    """

    def update(self, **kwargs: Any) -> Register:
        """改变成员的值.

        Args:
            kwargs (Any): 改变寄存器成员的值.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"{self.__class__.__name__} 没有 {k} 成员")
            if k in ["name"]:
                self.update_name(v)
            else:
                setattr(self, k, v)
        return self


class Register(RegisterBase):
    """32 位寄存器"""

    def __hash__(self) -> int:
        """取得哈希码 这样寄存器就可以放到 set 里边了

        Returns:
            int: 哈希码
        """
        return hash(f"{self.name}, {self.index}")

    def release(self) -> None:
        """释放寄存器.

        - `Register.alias = ""`
        - `Register.value = 0`
        - `Register.used_by = set()`
        - `Register.as_arg = ""`
        - `Register.as_return = ""`

        """
        self.alias = ""
        self.value = 0
        self.used_by = set()
        self.as_arg = ""  # 不用做函数输入
        self.as_return = ""  # 不用做函数输出

    def __str__(self) -> str:
        result = self.name
        if self.alias:
            result += f"({self.alias})"
        result += f" = {self.value}, used by: {sorted(self.used_by)}"
        if self.as_arg:
            result += f", func_arg: {self.as_arg}"
        if self.as_return:
            result += f", func_return: {self.as_return}"
        return result

    def __repr__(self) -> str:
        return self.__str__()

    def replace_by(self, reg: Register) -> Register:
        """用 `reg` 的信息替换当前寄存器的信息.

        Args:
            reg (Register): 另一个寄存器

        Returns:
            Register: 修改过的当前寄存器.
        """
        self._name = reg.name
        self.alias = reg.alias
        self.index = reg.index
        self.value = reg.value
        return self

    @property
    def used_by_list(self) -> list[int]:
        """返回 `Register.used_by` 列表.

        Returns:
            list[int]: `Register.used_by` 列表.
        """
        return sorted(list(self.used_by))

    @property
    def first(self) -> int:
        """找到第一次使用的语句, i.e. `Register.used_by` 最小值.

        Raises:
            ValueError: 寄存器没有使用者.

        Returns:
            int: `Register.used_by` 最小值.
        """
        if not self.used_by:
            raise ValueError("找不到第一次使用的语句, 寄存器没有使用者.")
        return min(self.used_by)

    @property
    def last(self) -> int:
        """找到最后一次使用的语句, i.e. `Register.used_by` 最大值.

        Raises:
            ValueError: 寄存器没有使用者.

        Returns:
            int: `Register.used_by` 最大值.
        """
        if not self.used_by:
            raise ValueError("找不到最后一次使用的语句, 寄存器没有使用者")
        return max(self.used_by)
