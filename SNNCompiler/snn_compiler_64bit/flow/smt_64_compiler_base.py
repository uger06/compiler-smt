# pylint: disable=line-too-long
"""处理 SMT 64 位指令的常用函数和类
"""

from __future__ import annotations

import math
import stat
from dataclasses import dataclass, field
from functools import cached_property, partial, reduce
from typing import Callable, Union

from addict import Dict as AttrDict
from loguru import logger
from ortools.sat.python import cp_model

from ..frontend.stablehlo_parser import StableHLOProgram, StableHLOStatement
from ..common.smt_64_reg import Register64, RegisterCollection
from ..backend.smt_64_factory import SMT64Factory

@dataclass
class SMT64CompilerBase:
    """SMT 64-bit 编译器基础成员

    >>> import brainpy as bp
    >>> funcs = {"V": bp.neurons.LIF(256).derivative}
    >>> SMT64CompilerBase(funcs=funcs).stablehlo_program.func_body.return_statement.operands[0].name
    'V'
    >>> funcs = {"HindmarshRose": bp.neurons.HindmarshRose(256).derivative}
    >>> SMT64CompilerBase(funcs=funcs).stablehlo_program.func_body.return_statement.operands[0].name
    'HindmarshRose'
    >>> compiler = SMT64CompilerBase(funcs = {"V": bp.neurons.LIF(256).derivative})
    >>> compiler.regs[3]
    <R3 = 0, used by: []>
    >>> compiler.stablehlo_statements[4]
    [StableHLOStatement(reg_index=4, cmd='convert', operands=[<R1(t) = 0, used by: [-1, 4], func_arg: t>])]
    """

    funcs: dict[str, Callable]
    """需要编译的函数返回值名称和函数体
    """

    update_method: dict[str, str] = field(default_factory=dict)
    """输入参数的更新方法

    - acc: 累加
    - update: 直接更新

    e.g
    ```python
    {
        "V": "acc",
        "g1": "update",
        "g2": "update"
    }
    ```
    """

    regs: RegisterCollection = None
    """寄存器集合, 如果为 None 则在构造函数中赋值为 `self.smt_factory.regs`"""

    smt_factory: SMT64Factory = field(init=False)
    """SMT 语句生成器"""

    stmt_results: dict[int, Register64] = field(default_factory=AttrDict)
    """每一条 StableHLO 语句运算的结果. key 为语句的索引值, value 为结果寄存器
    """

    constants: dict[str, Union[int, float]] = field(default_factory=dict)
    """常数, e.g. V_th
    """

    def __post_init__(self) -> None:
        """构造函数

        >>> import brainpy as bp
        >>> compiler = SMT64CompilerBase(funcs = {"V": bp.neurons.LIF(256).derivative})
        """
        self.regs = self.regs or RegisterCollection()
        self.smt_factory = SMT64Factory(regs=self.regs)

    @cached_property
    def stablehlo_program(self) -> StableHLOProgram:
        """StableHLO 程序, 如果输入是函数组`JointEq`, 则合并所有函数体

        Returns:
            StableHLOProgram: StableHLO 程序
        """
        result = StableHLOProgram.load(self.funcs)
        # for i, return_name in enumerate(self.funcs):
        #     # 添加 name 属性, 用于记录结果寄存器的名称, e.g. name = "V"
        #     result.func_body.return_statement.operands[i].name = return_name
        for i, return_name in enumerate(self.funcs):
            ## TODO, uger, 20240510, return_name 可能是{delta_V,u,g1,g2} 不一定对齐{V,g1,g2}
            # 记录结果寄存器的名称, 区分 "delta_V" and "V"
            if result.func_body.return_statement.operands[i].name == 'V':
                result.func_body.return_statement.operands[i].name = return_name
        return result

    @cached_property
    def func_args(self) -> dict[int, Register64]:
        """函数参数编号到寄存器的映射, key 为参数的编号, value 为参数的寄存器,
        e.g. `{0: V, 1: I}`

        Returns:
            AttrDict[int, Register]: 函数参数编号到寄存器的映射, e.g. `{0: V, 1: I}`
        """
        result: dict[int, Register64] = {}

        for arg_name in self.stablehlo_program.func_arg_names.values():
            # if arg_name.startswith("I"):  # 所有以 I 开头的参数都强制改名为 I
            #     arg_name = "I"
            try:  # 之前编译过程使用过的参数寄存器需要被重用
                reg = next(r for r in self.regs.used_regs if arg_name in [r.alias, r.as_arg])
            except StopIteration:  # 分配参数寄存器, 之前使用的寄存器不可以被使用
                reg = self.regs.get_unused_reg().update(used_by={-1})
            reg.update(alias=arg_name, as_arg=arg_name)
            result[len(result)] = reg

        # 上一次使用的寄存器可以再次被中间变量使用
        for reg in self.regs.used_regs:
            if reg.used_by == {-3}:
                self.regs.release(reg)

        return result

    @cached_property
    def return_names(self) -> dict[int, str]:
        """返回所有输出变量的名字,
        key 为输出 StableHLO 语句编号, value 为输出变量名称,
        e.g. {8: "V"} 表示第 8 条语句的结果为函数结果 V

        Returns:
            dict[int, str]: 所有输出语句编号和对应变量名称
        """
        result: dict[int, str] = {}
        for opr in self.stablehlo_program.func_body.return_statement.operands:
            # opr.name 属性是 `StableHLOProgram.load` 方法添加的
            result[opr.value] = opr.name
        return result

    @cached_property
    def stablehlo_statements(self) -> dict[int, list[StableHLOStatement]]:
        """返回更新过操作数的 StableHLO 指令列表,

        `self.stmt_results`

        - 操作数从字符串转换为寄存器或常数

        Returns:
            dict[int, list[StableHLOStatement]]: IR 指令列表
        """
        result: dict[int, list[StableHLOStatement]] = {}
        for stmt_id, stmt in enumerate(self.stablehlo_program.statement_list):
            self.get_result_reg(stmt_id=stmt_id)  # 确保每一条语句都有一个结果寄存器
            operands = []
            for opr in stmt.operands:
                opr_type, opr_value = opr.split(":")
                opr_value = opr_value.strip()
                if opr_type == "arg_index":
                    opr_value = int(opr_value)
                    reg = self.func_args[opr_value]
                    reg.used_by.add(stmt_id)
                    operands.append(reg)
                elif opr_type == "constant":
                    opr_value = float(opr_value)
                    operands.append(opr_value)
                elif opr_type == "reg_index":
                    opr_value = int(opr_value)
                    reg = self.stmt_results[opr_value]
                    reg.used_by.add(stmt_id)
                    operands.append(reg)
                else:
                    raise NotImplementedError(f"暂不支持 {opr_type = }, {opr_value = }")
                stmt.operands = operands
            result[stmt_id] = [stmt]
        return result

    def get_result_reg(self, stmt_id: int) -> Register64:
        """返回 `stmt_id` 语句的结果寄存器,
        如果未分配则为其分配一个虚拟寄存器, 并记录在 `self.stmt_results`

        Returns:
            Register64: 结果寄存器
        """
        if stmt_id in self.stmt_results:
            return self.stmt_results[stmt_id]

        result = self.regs.get_dummy_reg()
        result.used_by.add(stmt_id)

        if stmt_id in self.return_names:
            result.as_return = self.return_names[stmt_id]
            result.alias = self.return_names[stmt_id]
            return_index = next(i for i, v in enumerate(self.return_names.values()) if v == result.as_return)
            # 所有语句执行完之后, 再逐一保存函数结果, 所以也需要被使用一次
            result.used_by.add(return_index + len(self.stablehlo_program.func_body.statements))

        self.stmt_results[stmt_id] = result
        return result

    def replace_result_reg(self, old_reg: Register64, new_item: Union[Register64, float] = None) -> None:
        """使用 `new_item` 更新结果寄存器和语句中的操作数与结果寄存器

        - 更新 `self.stmt_results`
        - 更新 `self.stablehlo_statements`

        Args:
            old_reg (Register64): 旧寄存器
            new_item (Union[Register64, float]): 新的寄存器或者常数
        """
        if not isinstance(new_item, (Register64, float)):
            raise TypeError(f"不支持 {type(new_item) = }")

        # 更新操作数寄存器
        replaced = False
        statements = reduce(lambda a, b: a + b, list(self.stablehlo_statements.values()))
        for sh_stmt in statements:
            if sh_stmt.reg_index == old_reg.index and isinstance(new_item, Register64):
                sh_stmt.reg_index = new_item.index
                replaced = True
            if old_reg not in sh_stmt.operands:
                continue
            for opr_index, opr in enumerate(sh_stmt.operands):
                if old_reg == opr:
                    sh_stmt.operands[opr_index] = new_item
                    replaced = True

        # 更新结果寄存器
        for stmt_id, reg in self.stmt_results.items():
            if old_reg != reg:
                continue
            self.stmt_results[stmt_id] = new_item
            replaced = True

        if not replaced:
            raise ValueError(f"没有找到需要替换的寄存器 {old_reg = } {new_item = }")

        if isinstance(new_item, Register64):
            new_item.used_by = old_reg.used_by | new_item.used_by

        self.regs.release(old_reg)
