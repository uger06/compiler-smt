# pylint: disable=line-too-long
"""SMT 64 位指令编译器
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Union

from addict import Dict as AttrDict
from loguru import logger
from ortools.sat.python import cp_model

from ..common.asm_IEEE754 import IBinary, IEEE754
from ..backend.smt_64_stmt import SMT64, NOP
from ..common.smt_64_reg import Register64, RegisterCollection
from ..frontend.stablehlo_parser import StableHLOProgram, StableHLOStatement

from .smt_64_compiler_base import SMT64CompilerBase


@dataclass
class SMT64Compiler(SMT64CompilerBase):
    """SMT 64-bit 编译器

    1. 从函数得到的解析过的 stableHLO, 如果输入是函数组`JointEq`, 则合并所有函数体
    2. 对于每一条 `self.stablehlo_statements[stmt_id]`
        - 修改结果寄存器 `self.stmt_results[stmt_id]`
        - 修改 `cmd`, e.g. `divide` 变成 `multiply`
        - 修改 `operands`, 保存寄存器对象和数值
        - 修改 `reg_index`, 保存结果寄存器对象

    Example:
        >>> import brainpy as bp
        >>> funcs = {"V": bp.neurons.LIF(256).derivative}
        >>> compiler = SMT64Compiler(funcs=funcs)
        >>> compiler.compile()
        [R2 = 0.00 - R0, NOP, NOP, NOP, NOP, R2 = 0.00 + R2, NOP, NOP, NOP, NOP, R3 = 1.00 * R1, NOP, NOP, NOP, NOP, R2 = R2 + R3, NOP, NOP, NOP, NOP, R2 = 0.10 * R2, NOP, NOP, NOP, NOP]

    """

    _compiled: bool = False
    """是否已经编译"""

    _stablehlo_statements_updated: bool = False
    """是否已经更新了 `self.stablehlo_statements`"""

    smt64_results: list[SMT64] = field(default_factory=list)
    """编译结果, SMT 语句"""

    def check_type(self, operands: list, types: list[type], reporter: str) -> None:
        """检查类型是否支持

        Args:
            operands (list): 操作数
            types (list[type]): 支持的类型
            reporter (str): 报告者
        """

        for i, (opr, opr_type) in enumerate(zip(operands, types)):
            if isinstance(opr, opr_type):
                continue
            raise TypeError(f"{reporter}: 不支持非 {opr_type}, {type(operands[i]) = }")

    def update_stablehlo_statements(self) -> None:
        """更新 `self.stablehlo_statements`

        >>> import brainpy as bp
        >>> funcs = {"V": bp.neurons.LIF(256).derivative}
        >>> compiler = SMT64Compiler(funcs=funcs)
        >>> compiler.update_stablehlo_statements()
        >>> compiler._stablehlo_statements_updated
        True
        >>> compiler.stablehlo_statements[4]
        []

        """
        if self._stablehlo_statements_updated:
            return

        self._stablehlo_statements_updated = True

        for stmt_id, stmts in self.stablehlo_statements.items():
            stmt = stmts[0]
            old_reg = self.stmt_results[stmt_id]
            stmt.reg_index = old_reg.index
            operands = stmt.operands
            if stmt.cmd == "negate":
                self.check_type(operands, [Register64], stmt.cmd)
                stmt.cmd = "subtract"
                stmt.operands = [0, operands[0]]
            elif stmt.cmd == "convert":
                self.check_type(operands, [(Register64, float)], stmt.cmd)
                self.stablehlo_statements[stmt_id] = []  # 删除 convert 语句
                self.replace_result_reg(old_reg=old_reg, new_item=operands[0])
            elif stmt.cmd == "constant":
                self.check_type(operands, [float], stmt.cmd)
                self.replace_result_reg(old_reg=old_reg, new_item=operands[0])
                self.stablehlo_statements[stmt_id] = []  # 删除 constant 语句
            elif stmt.cmd in ["add", "multiply", "subtract"]:
                self.check_type(operands, [(Register64, float), (Register64, float)], stmt.cmd)
            elif stmt.cmd in ["exponential", "log"]:
                self.check_type(operands, [Register64], stmt.cmd)
            elif stmt.cmd == "divide":
                self.check_type(operands, [(Register64, float), (Register64, float)], stmt.cmd)
                if isinstance(operands[0], float) and isinstance(operands[1], float):  # 纯常数运算
                    self.replace_result_reg(old_reg=old_reg, new_item=operands[0] / operands[1])
                    self.stablehlo_statements[stmt_id] = []  # 删除 divide 语句
                elif isinstance(operands[0], Register64) and isinstance(operands[1], float):
                    stmt.cmd = "multiply"
                    stmt.operands = [operands[0], 1.0 / operands[1]]  # 寄存器/常数, 修正为乘法
                # 被除数若是寄存器, 28nm可以支持运算
                # else:
                #     raise NotImplementedError(f"暂不支持被除数是寄存器 {stmt.cmd} {operands[0] = } {operands[1] = }")
            elif stmt.cmd == "power":
                self.update_power(stmt_id, stmt)
            else:
                raise NotImplementedError(f"暂不支持指令 {stmt.cmd}")

        return

    def update_power(self, stmt_id: int, stmt: StableHLOStatement) -> None:
        """更新 power 语句"""

        operands = stmt.operands
        self.check_type(operands, [(Register64, float), float], stmt.cmd)
        self.stablehlo_statements[stmt_id] = []

        # 常数指数 2^1
        if isinstance(operands[0], float):
            self.replace_result_reg(old_reg=self.stmt_results[stmt_id], new_item=operands[0] ** operands[1])
            return
        
        # x^1
        if operands[1] == 1:
            self.replace_result_reg(old_reg=self.stmt_results[stmt_id], new_item=operands[0])
            return

        if int(operands[1]) != operands[1]:
            raise ValueError(f"power: 暂不支持浮点数指数 {operands[1] = }")

        for _ in range(int(operands[1]) - 2):
            self.stablehlo_statements[stmt_id].append(
                StableHLOStatement(
                    reg_index=operands[0].index,
                    cmd="multiply",
                    operands=[operands[0], operands[0]],
                )
            )
        self.stablehlo_statements[stmt_id].append(
            StableHLOStatement(
                reg_index=self.stmt_results[stmt_id].index,
                cmd="multiply",
                operands=[operands[0], operands[0]],
            )
        )

    def get_return_reg(self, return_name: str) -> Register64:
        """返回函数结果寄存器

        Args:
            return_name (str): 函数结果名称

        Returns:
            Register64: 函数结果寄存器
        """
        for stmt_id, name in self.return_names.items():
            if return_name == name:
                return self.stmt_results[stmt_id]
        raise ValueError(f"没有找到函数结果寄存器 {return_name = }")

    # pylint: disable-next=too-many-locals
    def merge_regs(self) -> None:
        """合并虚拟寄存器

        寄存器分配问题可以看作是一个约束编程问题,
        整个解空间可以看作为一个 2 维空间,
        一个维度是空闲寄存器的编号
        另一个维度是该寄存器的使用情况
        通过 ortools.sat.python.cp_model.CpModel 求解器求解

        - 使得寄存器的编号尽可能小
        - 不发生冲突
        """

        if not (regs_to_merge := self.regs.dummy_regs):
            return

        regs_to_merge: list[Register64]
        unused_regs = self.regs.unused_regs

        # region: 约束编程模型 model
        model = cp_model.CpModel()
        max_value = max(reg.last for reg in regs_to_merge)

        reg_ids = []  # 空闲寄存器编号
        usages = []  # 寄存器使用情况

        for i, reg in enumerate(regs_to_merge):
            reg_ids.append(
                model.NewIntVar(
                    0,
                    len(unused_regs) - 1,
                    f"real_reg_index_for_reg_{i}",
                )
            )
            usages.append(
                model.NewFixedSizeIntervalVar(
                    start=(reg_ids[-1] * max_value) + reg.first + 1,
                    size=reg.last - reg.first,
                    name=f"usage_{i}",
                )
            )

        model.Minimize(sum(reg_ids))  # 使得寄存器的编号尽可能小
        model.AddNoOverlap(usages)  # 不发生冲突
        # endregion: 约束编程模型 model

        # region: 约束编程求解
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        for status_str in ["unknown", "model_invalid", "feasible", "infeasible", "optimal"]:
            if status == getattr(cp_model, status_str.upper()):
                status_str = status_str.upper()
                break

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"求解失败: {status_str}.")
        # endregion: 约束编程求解

        for old_reg_id, real_reg_id in enumerate(reg_ids):
            new_reg_id = solver.Value(real_reg_id)
            old_reg = regs_to_merge[old_reg_id]
            new_reg = unused_regs[new_reg_id]
            # logger.warning(f"{old_reg.name} is replaced by {new_reg.name}")  ## uger
            self.replace_result_reg(old_reg=old_reg, new_item=new_reg)

    def compile(self) -> list[SMT64]:
        """得到 SMT64 语句

        >>> import brainpy as bp
        >>> funcs = {"V": bp.neurons.LIF(256).derivative}
        >>> compiler = SMT64Compiler(funcs=funcs)
        >>> compiler.compile()
        [R2 = 0.00 - R0, NOP, NOP, NOP, NOP, R2 = 0.00 + R2, NOP, NOP, NOP, NOP, R3 = 1.00 * R1, NOP, NOP, NOP, NOP, R2 = R2 + R3, NOP, NOP, NOP, NOP, R2 = 0.10 * R2, NOP, NOP, NOP, NOP]

        Returns:
            list[SMT64]: SMT 语句
        """

        if self._compiled:
            return self.smt64_results

        # stableHLO 变换成 SMT 64 语句支持的操作, 结果为虚拟寄存器
        self.update_stablehlo_statements()

        # 使用预定义的结果寄存器存储运算结果
        for return_id, name in self.return_names.items():
            if name not in ["g1", "g2"]:  ## 更新 g1 g2, 否则跳过
                continue                
            for used_reg in self.regs.used_regs:
                if used_reg.alias != name:    ## 跳过不一致的输出变量
                    continue
                if self.stmt_results[return_id] == used_reg:
                    continue
                self.replace_result_reg(old_reg=self.stmt_results[return_id], new_item=used_reg)
                break

        # 释放没被使用的输入寄存器
        for reg in self.regs.used_regs:
            if reg.used_by == {-1}:
                self.regs.release(reg)

        # 合并虚拟寄存器
        # NOTE: uger, 对虚拟寄存器进行合并，同时相当于占用了新的寄存器。
        self.merge_regs()

        # 结果寄存器添加名称
        # NOTE: uger, 此处对stmt返回结果的寄存器添加返回变量
        for stmt_id, name in self.return_names.items():
            self.stmt_results[stmt_id].as_return = name
            self.stmt_results[stmt_id].alias = name

        self.smt64_results = []

        for stmt_id, stmts in self.stablehlo_statements.items():
            for stmt in stmts:
                cmd_method = getattr(self.smt_factory, stmt.cmd, None)
                if cmd_method is None:
                    raise NotImplementedError(f"暂不支持指令 {stmt.cmd}")
                elif cmd_method in [self.smt_factory.add, 
                                    self.smt_factory.multiply, 
                                    self.smt_factory.subtract,
                                    self.smt_factory.divide]:
                    a, b = stmt.operands[:2]
                    if isinstance(b, float):
                        b = IEEE754(float(b))    #NOTE, uger, fix the bug of float operands
                    c = self.stmt_results[stmt_id]
                    self.smt64_results += cmd_method(a=a, b=b, c=c)
                elif cmd_method in [self.smt_factory.exponential, 
                                    self.smt_factory.log]:
                    a = stmt.operands[0]
                    c = self.stmt_results[stmt_id]
                    self.smt64_results += cmd_method(a=a, c=c)
                if stmt.cmd in ["add", "multiply", "subtract"]:
                    self.smt64_results.extend([NOP()] * 4)
                elif stmt.cmd in ["exponential", "log"]:
                    self.smt64_results.extend([NOP()] * 9)
                elif stmt.cmd in ["divide"]:
                    self.smt64_results.extend([NOP()] * 7)

        # 1. 通过 as_return 成员找到函数结果寄存器
        # 2. 更新其函数输入寄存器 (相同别名的寄存器)
        for result_reg in filter(lambda r: r.as_return is not None, self.regs.used_regs):
            if result_reg.as_return in ["delta_V", "V0", "V1"]:  # 跳过 V 寄存器
                continue
            for arg_reg in self.regs.used_regs:
                if result_reg == arg_reg:       ## 跳过自己
                    continue
                if arg_reg.alias == result_reg.as_return:
                    ## FIXME, uger
                    # self.smt64_results += self.smt_factory.move(src=result_reg, dst=arg_reg)
                    break

        if any(r.as_return == "delta_V" for r in self.regs.used_regs):
            update_v = f"""
            1: R_delta_V = R_V + R_delta_V
            2: NOP
            3: NOP
            4: NOP
            5: NOP
            6: R_0 = {self.constants['V_th']} - R_delta_V  // R_0 = V_th - V
            7: NOP
            8: NOP
            9: NOP
            10: NOP
            """
            self.smt64_results += SMT64.create_from_expr(update_v, regs=self.regs)

        return self.smt64_results

    @classmethod
    def compile_all(
        cls,
        funcs: dict[str, Callable],
        constants: dict[str, float] = None, 
        predefined_regs: dict[str, str] = None,
        update_method: dict[str, str] = None,
    ) -> tuple[SMT64Compiler, SMT64Compiler, list[SMT64]]:
        """编译所有函数.

        0. 如果有运算 I 的函数, 使用一个编译器对象编译此函数
        1. 使用一个编译器对象编译其他函数

        Example:
            >>> import brainpy as bp
            >>> funcs = {"V": bp.neurons.LIF(
            ...     256,
            ...     V_rest=-52,
            ...     V_th=-67,
            ...     V_reset=-22,
            ...     tau=31,
            ...     tau_ref=77,
            ...     method="exp_auto",
            ...     V_initializer=bp.init.Normal(-55.0, 2.0),
            ... ).derivative}
            >>> i_compiler, v_compiler, statements = SMT64Compiler.compile_all(funcs=funcs)
            >>> statements[0]
            R0 = 81.00 * R4

        Args:
            funcs (dict[str, Callable]): 需要编译的函数返回值名称和函数体,
                如果返回值名称第一个字母是 I, 则会被认为是 I 函数并改名为 I
            constants (dict[str, float]): 常数, e.g. {"V_th": -50}
            predefined_regs (dict[str, str], Optional): 预定义的寄存器, e.g.
                {"V": "R3"}. 默认为 None.

        Returns:
            tuple[SMT64Compiler, SMT64Compiler, list[SMT64]]: I 编译器, V 编译器, SMT 64 程序
        """

        predefined_regs = predefined_regs or {}  # 参数或结果名称到寄存器的映射, e.g. {"V": "R3"}

        # region: 创建编译器对象
        v_compiler = cls(
            funcs={k: v for k, v in funcs.items() if not k.startswith("I")},
            update_method=update_method,
        )
        compilers: list[SMT64Compiler] = [v_compiler]

        if i_func_names := [k for k in funcs if k.startswith("I")]:
            i_compiler = cls(
                funcs={k: funcs[k] for k in i_func_names},
                update_method=update_method,
            )
            compilers.insert(0, i_compiler)
        else:
            i_compiler = None
        # endregion: 创建编译器对象


        smt64_results: list[SMT64] = []

        used_regs: set[Register64] = set()  # 占用的寄存器的名字

        for compiler in compilers:
            for reg in used_regs:  # 上一个编译器使用的寄存器
                new_reg = compiler.regs[reg.index]
                new_reg.update(**reg.__dict__)
                new_reg.used_by = {-3}

            compiler.constants = constants

            # 预定义的寄存器
            for reg_alias, reg_name in predefined_regs.items():
                compiler.regs[reg_name].alias = reg_alias
                compiler.regs[reg_name].used_by |= {-2}

            smt64_results += compiler.compile()
            used_regs = set(compiler.regs.used_regs)

        return i_compiler, v_compiler, smt64_results
