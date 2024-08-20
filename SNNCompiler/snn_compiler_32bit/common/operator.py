"""控制字段
"""
from enum import Enum


class Operator(Enum):
    """控制字段, 不涉及位宽"""

    NOP: int = 0
    """空操作
    """

    CALCU: int = 1
    """计算

    - 32-bit 多路并行加法单元的两个源操作数和一个目的操作数选择: A1 A2 S
    - 32-bit 多路并行乘法单元的两个源操作数和一个目的操作数选择: M1 M2 P
    - 一条 Calcu 指令可以同时执行一次多路并行加法和一次多路并行乘法.
    """

    SRAM_LOAD: int = 2
    """读存储

    将存储在 BRAM 缓存中的目的神经元的独享参数载入到独享常量/独享变量寄存器中
    """

    SRAM_SAVE: int = 3
    """写存储

    将独享常量/独享变量寄存器中的神经元独享参数存回到 BRAM 缓存中
    """

    V_SET: int = 4
    """膜电位更新

    令通过判断源操作数 A1 的值以及内部相关逻辑 (神经元是否处于不应期)
    来决定是否将源操作数 A2 的值赋给目的操作数
    该指令一般用来做膜电位更新操作, 其中 A1 和 A2 为源操作数选择信号, S 为目的操作数选择信号

    Example:

    `V_SET: A1=temp0, A2=V_reset, S=V`

    根据 temp0 的值的最高位是否为负以及是否处于不应期(硬件电路实现), 判断是否将 V_reset 的值赋值给 V.
    """

    SPIKE: int = 5
    """脉冲发放

    通过判断源操作数 A1 的值以及内部相关逻辑 (神经元是否处于不应期) 来决定是否发放脉冲
    """

    END: int = 6
    """仿真结束

    标志当前计算结束, 令 SMT 地址跳转到首地址开始下一次计算, 控制 SMT 存储器的地址寄存器的更新
    """

    def __str__(self) -> str:
        """输出 `f"{self.name}({self.value})"`

        Returns:
            str: `f"{self.name}({self.value})"`
        """
        return f"{self.name}({self.value})"

    def __repr__(self) -> str:
        """输出 `f"{self.name}({self.value})"`

        Returns:
            str: `f"{self.name}({self.value})"`
        """
        return self.__str__()
