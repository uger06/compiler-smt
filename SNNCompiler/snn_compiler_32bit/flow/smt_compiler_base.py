"""stablehlo to SMT
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set

from addict import Dict as AttrDict

from ..common.register_collection import RegisterCollection

from ..backend.smt import SMT, Register
from ..backend.smt_factory import SMTFactory
from ..common import Number


@dataclass
class SMTCompilerBase:  # pylint: disable=too-many-instance-attributes
    """SMT 编译器基类. 包含基本成员和方法."""

    func: Dict[str, Callable]
    """需要编译的函数返回值名称和函数体.
    """

    V_thresh: Register = None
    """必须存在的 V_thresh 寄存器.
    """

    V_reset: Register = None
    """必须存在的 V_reset 寄存器.
    """

    # region: 记录 I 函数运算使用的成员, 其他函数运算不能使用这些寄存器作为函数输入.
    is_i_func: bool = False
    """`func` 是 I 运算.
    """

    i_reg_name: str = ""
    """I 参数的寄存器名字, e.g. `R3`.
    """

    used_arg_names: Dict[str, str] = field(default_factory=AttrDict)
    """使用的输入变量名称到寄存器名称的映射, e.g. `{"I": "R2"}`
    """

    used_shared_regs: Set[Register] = field(default_factory=set)
    """使用的共享寄存器, e.g. {R3, R4}
    """
    # endregion: 记录 I 函数运算使用的成员, 其他函数运算不能使用这些寄存器作为函数输入.

    update_method: Dict[str, str] = None
    """参数的更新方法, 默认累加, 支持的方法:

    - acc: 累加
    - update: 直接更新

    e.g.

    ```python
    {
        "V": "acc",
        "g1": "update",
        "g2": "update"
    }
    ```
    """

    smt_factory: SMTFactory = None
    """SMT 生成器对象
    """

    preload_constants: Set[Register] = field(default_factory=set)
    """常数寄存器, 需要预先加载到 `property.bin`.
    `ZERO_REG` 和 `ONE_REG` 会在构造函数里添加到 `preload_constants`.
    """

    smt_info_str: str = ""
    """未优化的 SMT 语句.
    """

    _reg_results: Dict[int, Register] = field(default_factory=AttrDict)
    """每一条 IR 语句运算的结果. 编译使用的成员.
    """

    _smt_results: Dict[int, List[SMT]] = field(default_factory=AttrDict)
    """编译结果, SMT 语句. 编译使用的成员.
    """

    _compiled: bool = False
    """已经编译过了. 编译使用的成员.
    """

    result_bits: int = 3
    """结果寄存器宽度, 3 或 4.
    """

    @property
    def regs(self) -> RegisterCollection:
        """当前编译使用的寄存器库.

        Returns:
            RegisterCollection: 当前编译使用的寄存器库.
        """
        return self.smt_factory.regs

    def __post_init__(self) -> None:
        """构造函数"""

        self.update_method = self.update_method or {}
        self.smt_factory = self.smt_factory or SMTFactory(regs=RegisterCollection(result_bits=self.result_bits))

        self.preload_constants |= {
            self.regs.SR0.update(used_by={-2}, alias="V_reset"),
            self.regs.SR1.update(used_by={-2}, alias="T_refrac"),
        }
        self.preload_constants.add(self.regs.ZERO_REG)

    def add_constant_reg(self, value: Number) -> Register:
        """添加常数寄存器.

        - 如果常数已存在, 返回常数寄存器.
        - 如果常数不存在, 返回新构造的常数寄存器.

        Returns:
            Register: 常数寄存器
        """
        # 如果常数已经存在则返回其寄存器
        for reg in self.preload_constants:
            if reg.value == value:
                return reg

        # 如果常数为 0 则返回 ZERO_REG 寄存器
        if value == 0:
            result = self.regs.ZERO_REG
            self.preload_constants.add(result)
            return result

        # 如果常数为 1 则返回 ONE_REG 寄存器
        if value == 1:
            result = self.regs.ONE_REG
            self.preload_constants.add(result)
            return result

        # 添加新的常数寄存器
        result = self.regs.unused_shared_reg
        result.update(value=value, used_by={-2})
        self.preload_constants.add(result)
        return result
