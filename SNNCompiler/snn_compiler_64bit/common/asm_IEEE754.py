from __future__ import annotations

import re
import struct
from ctypes import Union
from dataclasses import dataclass
from enum import IntEnum


@dataclass
class IBinary:
    """二进制数值接口, 无符号

    - `dec_value` (int): 十进制数值, 默认值 0
    - `bin_width` (int): 二进制宽度, 默认值 32
    - `bin_value` (str): 二进制数值字符串, e.g. "000100"
    - `dec2bin` (int, int) -> str: 十进制数值转换成二进制数值
    - `bin2dec` (str, bool) -> int: 二进制数值转换成十进制数值
    """

    dec_value: int = 0
    """十进制数值
    """

    bin_width: int = 32
    """二进制宽度
    """

    def __int__(self) -> int:
        """返回十进制数值"""
        return self.dec_value

    def __post_init__(self) -> None:
        """构造函数"""

        if isinstance(self.dec_value, str):
            self.dec_value = self.bin2dec(self.dec_value)
            return

        if (is_int := isinstance(self.dec_value, int)) and (self.dec_value < 0):
            raise TypeError(f"dec_value 必须是正数 ({self.dec_value = })")

        if is_int and (self.dec_value >= (1 << self.bin_width)):
            raise ValueError(f"{self.dec_value = } 太大. 位宽 {self.bin_width = } 不够")

    @staticmethod
    def dec2bin(dec_value: int, bin_width: int = 32) -> str:
        """十进制数值转换成二进制数值

        Example:
            >>> IBinary.dec2bin(-31, 32)
            Traceback (most recent call last):
            ...
            TypeError: dec_value 必须是正数 (dec_value = -31)
            >>> IBinary.dec2bin(32, 8)
            '00100000'
            >>> IBinary.dec2bin(32, 5)
            Traceback (most recent call last):
            ..
            ValueError: dec_value = 32 太大. 位宽 bin_width = 5 不够
            >>> IBinary.dec2bin(16, 5)
            '10000'

        Args:
            dec_value (int): 十进制数值
            bin_width (int, optional): 二进制数值的宽度. 默认值 32

        Raises:
            TypeError: 如果输入是负数
            ValueError: 如果输入数值超过位宽的范围

        Returns:
            str: 二进制数值
        """
        if dec_value < 0:
            raise TypeError(f"dec_value 必须是正数 ({dec_value = })")

        if dec_value >= (1 << bin_width):
            raise ValueError(f"{dec_value = } 太大. 位宽 {bin_width = } 不够")

        return f"{dec_value:0{bin_width}b}"

    @staticmethod
    def bin2dec(bin_value: str) -> int:
        """二进制数值转换成十进制数值, 无符号

        Example:
            >>> IBinary.bin2dec("0b00000000000000000000000000011111")
            31
            >>> IBinary.bin2dec("100001")
            33
            >>> IBinary.bin2dec("11")
            3

        Args:
            bin_value (str): 二进制数值

        Returns:
            int: 十进制数值
        """
        return int(bin_value, 2)

    @property
    def bin_value(self) -> str:
        """二进制数值, 将十进制数值转换为二进制数值

        Example:
            >>> IBinary(31).bin_value
            '00000000000000000000000000011111'
            >>> IBinary(39).bin_value
            '00000000000000000000000000100111'
            >>> IBinary(31, bin_width=8).bin_value
            '00011111'
            >>> IBinary(33, bin_width=8).bin_value
            '00100001'
            >>> f = IBinary()
            >>> f.bin_value = "0b00000000000000000000000000011111"
            >>> f.dec_value
            31
            >>> f.bin_value = "100001"
            >>> f.dec_value
            33

        Returns:
            str: 转换后的二进制数值
        """
        return self.dec2bin(self.dec_value, self.bin_width)

    @bin_value.setter
    def bin_value(self, value: str) -> None:
        """更新二进制数值, 同时更新十进制数值

        Example:
            >>> f = IBinary()
            >>> f.bin_value = "0b00000000000000000000000000011111"
            >>> f.dec_value
            31
            >>> f.bin_value = "100001"
            >>> f.dec_value
            33

        Args:
            bin_value(str): 二进制数值
            signed(bool, optional): 二进制数值首位是否是符号位. 默认值 False
        """
        self.dec_value = self.bin2dec(value)

    def __str__(self) -> str:
        return str(self.dec_value)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self.dec_value}:{self.bin_value}>"


class IEEE754(IBinary):
    """IEEE 754 格式的 32 位浮点数

    - 整数不转化为 IEEE 754 格式的整数
    - 浮点数转化为 IEEE 754 格式的整数
    - 二进制数转化为整数, 不转化为 IEEE 754 格式的整数

    Example:
        >>> IEEE754(1).dec_value
        1
        >>> IEEE754("101").dec_value
        5
        >>> IEEE754(-1).dec_value
        Traceback (most recent call last):
        ...
        ValueError: value = -1 不是正数
        >>> IEEE754(1.0).dec_value
        1065353216
        >>> IEEE754(-1.0).dec_value
        3212836864
        >>> IEEE754("00111111100000000000000000000000").dec_value
        1065353216
        >>> IEEE754.float_to_ieee754(1.0)
        1065353216
        >>> IEEE754.float_to_ieee754(1)
        1065353216
        >>> IEEE754.ieee754_to_float(1065353216)
        1.0
        >>> -IEEE754(1.0)
        <IEEE754:3212836864:-1.00>
        >>> -(-IEEE754(1.0))
        <IEEE754:1065353216:1.00>
        >>> (-IEEE754(1.0)).float_value
        -1.0
        >>> -IEEE754(1)
        <IEEE754:2147483649:-0.00>
        >>> IEEE754(1).float_value
        1.401298464324817e-45
        >>> (-IEEE754(1)).float_value
        -1.401298464324817e-45
    """

    raw_value: Union[int, float, "IEEE754"]
    """原始值, 二进制字符串按照整数处理"""

    def __init__(self, value: Union[str, int, float, "IEEE754"]) -> None:
        """构造函数

        - 整数不转化为 IEEE 754 格式的整数
        - 浮点数转化为 IEEE 754 格式的整数
        - 二进制数转化为整数, 不转化为 IEEE 754 格式的整数

        Args:
            value (Union[str, int, float, IEEE754]): 整数, 浮点数, 二进制数, IEEE 754 格式的整数
                - IEEE754: IEEE 754 格式的浮点数
                - int: 整数, 不转化为 IEEE 754 格式的整数, 不能是负数
                - float: 浮点数, 转化为 IEEE 754 格式的整数
                - str: 无符号二进制数, 不转化为 IEEE 754 格式的整数
        """
        self.raw_value = value
        if isinstance(value, IEEE754):  # IEEE754 浮点数
            dec_value = value.dec_value
        elif isinstance(value, int):  # 整数, 不转化为 IEEE 754 格式, 不能是负数
            if value < 0:
                raise ValueError(f"{value = } 不是正数")
            dec_value = value
        elif isinstance(value, float):  # 浮点数需要转换成整数
            dec_value = self.float_to_ieee754(value)
        elif isinstance(value, str):  # 二进制需要转换成整数
            if not re.match(r"(0b)?[01]+", value):
                raise ValueError(f"{value = } 不是二进制数值")
            dec_value = IBinary.bin2dec(value)
            self.raw_value = int(value, 2)
        else:
            raise TypeError(f"{value = } 不是 IEEE754, int, float 或 str 类型")

        super().__init__(dec_value=dec_value, bin_width=32)

    @property
    def float_value(self) -> float:
        """浮点数值"""
        return self.ieee754_to_float(self.dec_value)

    @float_value.setter
    def float_value(self, value: Union[int, float]) -> float:
        """改变浮点数值, 同时改变原始值"""
        self.raw_value = value
        self.dec_value = self.float_to_ieee754(value)

    @staticmethod
    def float_to_ieee754(x: Union[int, float]) -> int:
        """将整数或浮点数转换为 IEEE 754 格式的 32 位整数

        Example:
            >>> IEEE754.float_to_ieee754(1.0)
            1065353216

        Args:
            x (Union[int, float]): 整数或浮点数

        Returns:
            int: 整数
        """
        return struct.unpack("<I", struct.pack("<f", x))[0]

    @staticmethod
    def ieee754_to_float(x: int) -> float:
        """将 IEEE 754 格式的整数转换为浮点数

        Example:
            >>> IEEE754.ieee754_to_float(1065353216)
            1.0

        Args:
            x (int): IEEE 754 格式的整数

        Returns:
            float: 浮点数
        """

        return struct.unpack("<f", struct.pack("<I", x))[0]

    def __neg__(self) -> IEEE754:
        """取负值, 原始值也取负值

        Example:
            >>> -IEEE754(1.0)
            <IEEE754:3212836864:-1.00>
            >>> -IEEE754(1.0).raw_value
            -1.0
            >>> -IEEE754(3)
            <IEEE754:2147483651:-0.00>
            >>> -IEEE754(3).raw_value
            -3
            >>> IEEE754(3)
            <IEEE754:3:0.00>
            >>> IEEE754(3).raw_value
            3

        Returns:
            IEEE754: IEEE 754 格式的负值
        """
        self.raw_value = -self.raw_value  # 最终的 IEEE754 对象原始值肯定不是 IEEE754 对象
        return IEEE754(-self.ieee754_to_float(self.dec_value))

    def __repr__(self) -> str:
        """返回字符串

        Example:
            >>> IEEE754(1.0)
            <IEEE754:1065353216:1.00>
            >>> IEEE754(1)
            <IEEE754:1:0.00>
        """
        return f"<{self.__class__.__name__}:{self.dec_value}:{self.float_value:.2f}>"


__all__ = [
    "IBinary",
    "IEEE754",
]
