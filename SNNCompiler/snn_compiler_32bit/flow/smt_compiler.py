"""stablehlo to SMT
"""
from __future__ import annotations

from functools import reduce
from typing import Callable, Dict, List

from addict import Dict as AttrDict
from ortools.sat.python import cp_model

from ..backend.smt import SMT, Operator, Register
from ..common import Number
from .smt_compiler_base_with_methods import SMTCompilerBaseWithMethods


class SMTCompiler(SMTCompilerBaseWithMethods):
    """SMT 编译器"""

    def get_stmt_id_by_reg(self, reg: Register) -> int:
        """根据结果寄存器对象, 返回 IR 语句的索引值.

        Args:
            reg (Register): 结果寄存器.

        Returns:
            int: IR 语句的索引值.
        """
        for result, result_reg in self._reg_results.items():
            if result_reg == reg:
                return result
        for arg_reg in self.func_args.values():
            if arg_reg == reg:
                return -1
        raise RuntimeError(f"{reg} 不是结果寄存器也不是函数输入寄存器.")

    def get_used_by_others(self, stmt_id: int, reg: Register) -> set[int]:
        """返回使用寄存器的其他语句.

        Args:
            stmt_id (int): 当前语句索引.
            reg (Register): 寄存器对象.

        Returns:
            set[int]: 使用寄存器的其他语句索引
        """
        return reg.used_by - {-1, -2, stmt_id}

    def get_negated(self, stmt_id: int, opr: Register) -> Register:
        """得到取负值的运算结果. 具体方法参见 `IRCmd.negate`.

        Args:
            stmt_id (int): 当前 IR 语句索引.
            opr (Register): 操作数寄存器.

        Returns:
            Register: 负数结果寄存器. `used_by` 已更新
        """
        # 找到使用正值的语句, 不包括当前语句.
        used_by_others = self.get_used_by_others(stmt_id=stmt_id, reg=opr)

        # 根据 V 使用的寄存器得到正值寄存器和负值寄存器
        r6 = self.regs.pos_reg
        r6_neg = self.regs.neg_reg

        if opr in self.preload_constants:  # 常数的负数
            sr_x_neg = self.add_constant_reg(-(opr.value))
            sr_x_neg.used_by |= {stmt_id}
            sr_x_neg.alias = f"-{opr.short}"

            if (not used_by_others) and (opr not in [self.regs.SR0, self.regs.SR1]):
                sr_x_neg.alias = f"{-opr.value}"
                opr.release()
                self.preload_constants.discard(opr)

            return sr_x_neg

        if opr.as_arg:  # 输入的负数
            self._smt_results[stmt_id] += self.smt_factory.move(src=opr, dst=r6)  # r6 = opr
            return r6_neg

        # 运算结果的负数
        # 找到之前的运算语句
        # ir_stmt_id = -1 则为函数输入
        ir_stmt_id = self.get_stmt_id_by_reg(opr)

        # 1. 将之前的运算结果保存在正值寄存器, e.g. `R6`
        if ir_stmt_id == -1:  # 函数输入的负数
            # 在当前 SMT 语句块将其复制 (R6 = opr + 0) 到 `R6` 取负值.
            self._smt_results[stmt_id] += self.smt_factory.move(src=opr, dst=r6)
        else:  # 运算语句的结果
            # 如果有运算, 则直接改变运算结果寄存器为 `R6`.
            # 先不改变结果寄存器.
            for ir_stmt in reversed(self._smt_results[ir_stmt_id]):
                if ir_stmt.op == Operator.CALCU:
                    ir_stmt.reg_result = r6
                    break
            else:
                # 如果没有运算比如 convert
                # 则在运算语句所在的语句块将其复制到 `R6` 取负值.
                self._smt_results[ir_stmt_id] += self.smt_factory.move(src=opr, dst=r6)

        sr_x_neg = self.regs.unused_dummy_reg  # 负值结果
        sr_x_neg.used_by.add(stmt_id)
        if ir_stmt_id > -1:
            sr_x_neg.used_by.add(ir_stmt_id)

        if (not used_by_others) or (used_by_others == set([ir_stmt_id])):  # 没有其他语句使用正值
            # 2. 添加 SMT 语句: SR_X_NEG = R6_NEG + ZERO_REG
            # 不能直接使用 r6_neg 因为可能会被其他语句占用
            self._smt_results[ir_stmt_id] += self.smt_factory.move(src=r6_neg, dst=sr_x_neg)

            # 3. 更新运算结果寄存器为 SR_X_NEG
            self._reg_results[ir_stmt_id] = self.regs.NONE_REG  # 没有其他运算使用这个结果

            # 5. 释放 opr
            opr.release()
            return sr_x_neg

        # 有其他语句使用正值
        # 2. 添加 SMT 语句: SR_X_NEG = R6_NEG + ZERO_REG; SR_X = R6 * ONE_REG
        sr_x = self.regs.unused_dummy_reg  # 正值, 用来代替 opr
        sr_x.used_by = opr.used_by  # 使用 SR_X 代替 opr

        # SR_X_NEG = R6_NEG + ZERO_REG; SR_X = R6 * ONE_REG
        self._smt_results[ir_stmt_id] += self.smt_factory.add_multiply(
            a1=r6_neg,
            a2=0,
            m1=r6,
            m2=1,
            s=sr_x_neg,
            p=sr_x,
        )

        # 3. 更新运算结果寄存器为 SR_X
        self._reg_results[ir_stmt_id] = sr_x

        # 4. 使用 SR_X 代替 opr
        for used_stmt_id in opr.used_by:  # 每一条 IR 语句
            if used_stmt_id < 0:  # 跳过函数输入和保留常数
                continue
            for smt_cmd in self._smt_results[used_stmt_id]:  # IR 语句对应的一组 SMT 语句
                smt_cmd: SMT
                smt_cmd.update_operand(old_reg=opr, new_reg=sr_x)  # 更新寄存器

        return sr_x_neg

    def cmd_constant(self, stmt_id: int) -> Register:
        """常数命令.

        操作数和结果是同一个寄存器对象.
        不记录占用, 因为肯定会被其它语句占用.

        Args:
            stmt_id (int): IR 语句索引.

        Returns:
            Register: 结果寄存器.
        """
        # 常数寄存器输入已经在 `SMTCompilerBase.ir_stmts` 中
        # 保存在 `SMTCompilerBase.preload_constants`.
        # 常数命令的操作数和结果是同一个寄存器对象
        # 释放占位符
        reg = self.ir_stmts[stmt_id].operands[0]
        self.update_result_reg(stmt_id=stmt_id, reg=reg)

        # 不记录占用, 因为肯定会被其它语句占用.
        reg.used_by.discard(stmt_id)

        return reg

    def cmd_convert(self, stmt_id: int) -> Register:
        """类型转换命令.

        操作数和结果是同一个寄存器对象.
        不记录占用, 因为肯定会被其它语句占用.

        Args:
            stmt_id (int): IR 语句索引.

        Returns:
            Register: 结果寄存器.
        """
        return self.cmd_constant(stmt_id=stmt_id)

    def cmd_add(self, stmt_id: int) -> Register:
        """加法.

        Args:
            stmt_id (int): IR 语句索引.

        Returns:
            Register: 结果寄存器.
        """
        opr = self.ir_stmts[stmt_id].operands
        result = self._reg_results[stmt_id]
        self._smt_results[stmt_id] += self.smt_factory.add(a=opr[0], b=opr[1], c=result)
        return result

    def cmd_multiply(self, stmt_id: int) -> Register:
        """乘法.

        Args:
            stmt_id (int): IR 语句索引.

        Returns:
            Register: 结果寄存器.
        """
        opr = self.ir_stmts[stmt_id].operands

        # 乘以 0
        if self.regs.ZERO_REG in opr:
            self._reg_results[stmt_id] = self.regs.ZERO_REG
            return self.regs.ZERO_REG

        result: Register = self._reg_results[stmt_id]
        self._smt_results[stmt_id] += self.smt_factory.multiply(opr[0], opr[1], result)
        return result

    def cmd_power(self, stmt_id: int) -> Register:
        """指数.

        Args:
            stmt_id (int): IR 语句索引.

        Returns:
            Register: 结果寄存器.
        """
        opr = self.ir_stmts[stmt_id].operands

        if opr[1] not in self.preload_constants:
            raise NotImplementedError(f"暂不支持非常数次幂 {opr[1]}")

        if opr[1].value != int(opr[1].value):
            raise NotImplementedError(f"暂不支持非整数次幂 {opr[1].value}")

        opr[1].value = int(opr[1].value)

        if opr[1].value < 0:
            raise NotImplementedError(f"暂不支持负数次幂 {opr[1].value}")

        if opr[1].value == 0:
            self.update_result_reg(stmt_id=stmt_id, reg=self.regs.ONE_REG)
            return self.regs.ONE_REG

        if opr[1].value == 1:
            return self.cmd_convert(stmt_id)

        last_result = self.regs.unused_dummy_reg
        result = last_result
        last_result.used_by.add(stmt_id)

        self._smt_results[stmt_id] += self.smt_factory.multiply(a=opr[0], b=opr[0], c=result)
        for _ in range(opr[1].value - 2):
            result = self.regs.unused_dummy_reg
            result.used_by.add(stmt_id)
            self._smt_results[stmt_id] += self.smt_factory.multiply(a=last_result, b=opr[0], c=result)
            last_result = result

        self.update_result_reg(stmt_id=stmt_id, reg=result)
        return result

    def cmd_negate(self, stmt_id: int) -> Register:
        """取负值运算. 具体方法参见 `IRCmd.negate`.

        Args:
            stmt_id (int): IR 语句索引.

        Returns:
            Register: 结果寄存器.
        """

        opr = self.ir_stmts[stmt_id].operands

        if opr[0] == self.regs.V:  # 如果 V 取负值, 返回 V_NEG
            result = self.regs.V_NEG
        else:  # 其他情况返回生成的负值
            result = self.get_negated(stmt_id=stmt_id, opr=opr[0])

        self.update_result_reg(stmt_id=stmt_id, reg=result)
        return result

    def cmd_divide(self, stmt_id: int) -> Register:
        """除法

        - 除数为常数, 只被除法 IRCmd.divide 使用: 存储除数的倒数到共享寄存器
        - 除数为常数, 被除法 IRCmd.divide 和其他运算使用: 存储常数和除数的倒数到共享寄存器
        - 除数为运算结果: 不支持

        之后运行乘法并输出乘积到共享寄存器

        优化方向:

        - 如果除数只被类型转换 IRCmd.convert 使用, 那么不用存储
        - 如果除数只被取负值 IRCmd.negate 使用, 那么不用存储

        Args:
            stmt_id (int): IR 语句索引.

        Returns:
            Register: 结果寄存器.
        """

        opr = self.ir_stmts[stmt_id].operands

        if opr[1] not in self.preload_constants:
            raise NotImplementedError("暂只支持除以常数")

        # 如果除数没有被其他语句使用, 不保存除数
        used_by_others = self.get_used_by_others(stmt_id=stmt_id, reg=opr[1])

        # 记录除数的倒数
        divider = self.add_constant_reg(1 / (opr[1].value))
        divider.used_by.add(stmt_id)
        divider.alias = f"1/{opr[1].value}"

        if not used_by_others:  # 如果除数没有被其他语句使用, 不保存除数
            if opr[1] not in (self.regs.SR0, self.regs.SR1):
                divider.alias = f"1/{opr[1].value}"
                opr[1].release()
                self.preload_constants.discard(opr[1])

        # 运行乘法代替除法
        result = self._reg_results[stmt_id]
        self._smt_results[stmt_id] += self.smt_factory.multiply(a=opr[0], b=divider, c=result)
        return result

    def cmd_subtract(self, stmt_id: int) -> Register:
        """减法

        - 减数为常数, 只被减法 `IRCmd.subtract` 使用: 存储减数的负数到共享寄存器.
        - 减数为常数, 被减法 `IRCmd.subtract` 和其他运算使用: 存储常数和减数的负数到共享寄存器.
        - 减数为运算结果: 取负数.

        之后运行加法法并输出和到共享寄存器

        Args:
            stmt_id (int): IR 语句索引.

        Returns:
            Register: 结果寄存器.
        """

        opr = self.ir_stmts[stmt_id].operands

        if opr[1] == self.regs.V:
            result = self._reg_results[stmt_id]
            self._smt_results[stmt_id] += self.smt_factory.add(a=opr[0], b=self.regs.V_NEG, c=result)
            return

        # 负值
        sr_x_neg = self.get_negated(stmt_id=stmt_id, opr=opr[1])

        # 运行加法代替减法
        result = self._reg_results[stmt_id]
        self._smt_results[stmt_id] += self.smt_factory.add(a=opr[0], b=sr_x_neg, c=result)
        return

    def compile(self) -> None:
        """得到 SMT 语句和预加载常数."""

        if self._compiled:
            return

        if not self.used_arg_names:
            self.used_arg_names = {r.alias: r.name for r in self.func_args.values()}
        else:
            pass

        for stmt_id, stmt in enumerate(self.ir_stmts):
            cmd_method = getattr(self, f"cmd_{stmt.cmd}", None)
            if cmd_method is None:
                raise NotImplementedError(f"暂不支持指令 {stmt.cmd}")
            cmd_method(stmt_id)

        # 保存结果的第一句语句编号
        stmt_id = len(self.parsed_ir.func_body.statements)
        i_reg = None

        # 记录函数结果
        for ir_id, arg_name in self.return_names.items():
            # 在输入的输出, e.g. V,
            # 使用 self.update_method[arg_name], acc: 累加, update: 更新
            if arg_name in self.parsed_ir.func_arg_names.values():
                arg_reg: Register = next(r for r in self.func_args.values() if r.as_arg == arg_name)

                if self.is_i_func and (arg_name == "I"):
                    i_reg = arg_reg  # 记录 I 使用的寄存器

                update_method = self.update_method.get(arg_name, "acc")

                # 累加, e.g. V = dV + V
                # 结果保存到 arg_reg
                if update_method == "acc":
                    self._reg_results[stmt_id] = arg_reg
                    arg_reg.used_by.add(stmt_id)
                    self._smt_results[stmt_id] += self.smt_factory.add(
                        a=self._reg_results[ir_id],
                        b=arg_reg,
                        c=arg_reg,
                    )
                    stmt_id += 1
                elif update_method == "update":
                    # 更新, e.g. g1, g2
                    target_reg = self.regs.get_reg_by_alias(arg_name)
                    for stmt in reversed(self._smt_results[ir_id]):
                        if stmt.op != Operator.CALCU:
                            continue
                        old_reg = self._reg_results[ir_id]
                        stmt.update_regs(old_reg=old_reg, new_reg=target_reg)
                        old_reg.release()
                        self._reg_results[ir_id] = target_reg
                        break
                else:
                    raise NotImplementedError(f"暂不支持 {update_method = }")
                continue

            # 不在输入的输出, e.g. I, 替换 dummy 寄存器为 unused_arg_reg
            if self.update_method.get(arg_name, None) is not None:
                # 需要存在结果寄存器, 不然会多需要一个寄存器
                for reg in reversed(self.regs.valid_func_arg_regs):
                    if reg.used_by:
                        continue
                    if reg not in self.regs.valid_result_regs:
                        continue
                    self.regs.use_reg(reg)
                    arg_reg = reg
                    break
                else:
                    raise RuntimeError("找不到未被占用的可作为结果寄存器的函数参数寄存器")
            else:
                arg_reg = self.regs.unused_arg_reg

            arg_reg.update(alias=arg_name, used_by={ir_id})

            for smt in self._smt_results[ir_id]:
                smt: SMT
                if smt.op != Operator.CALCU:
                    continue
                if smt.s == self._reg_results[ir_id]:
                    smt.s = arg_reg
                if smt.p == self._reg_results[ir_id]:
                    smt.p = arg_reg
            if self.is_i_func and (arg_name == "I"):
                i_reg = arg_reg  # 记录 I 使用的寄存器

        if self.is_i_func:
            self._compiled = True
            self.i_reg_name = i_reg.name  # 记录 I 使用的寄存器
            return

        # Delta V = R0 = V_thresh - V
        self._smt_results[stmt_id] += self.smt_factory.add(
            a=self.V_thresh,
            b=self.regs.V_NEG,
            c=self.regs.R0,
        )

        # # BUG, uger, for debug
        # self._smt_results[stmt_id] += self.smt_factory.multiply(
        #     a=self.regs.SR4,
        #     b=self.regs.V,
        #     c=self.regs.R3,
        # )

        # 根据 Delta V 设置 V = V_reset
        self._smt_results[stmt_id] += self.smt_factory.v_set(delta_v=self.regs.R0, v_reset=self.V_reset)

        # 根据 Delta V 发出激励
        self._smt_results[stmt_id] += self.smt_factory.spike(self.regs.R0)

        self._smt_results[stmt_id] += self.smt_factory.sram_save()
        self._smt_results[stmt_id] += self.smt_factory.end()

        self._compiled = True
        return

    def optimize(self) -> Dict[Register, Register]:
        """优化寄存器使用.

        Returns:
            Dict[Register, Register]: 合并映射.
        """
        reg_map = {}
        self.try_use_result_reg()
        reg_map = self.merge_regs()
        for old_reg, new_reg in reg_map.items():
            old_reg: Register
            old_reg.replace_by(new_reg)
        return reg_map

    def merge_regs(self) -> Dict[Register, Register]:
        """合并虚拟寄存器.

        Returns:
            Dict[Register, Register]: 合并映射.
        """

        if not (regs_to_merge := self.regs.dummy_regs):
            return {}

        result: AttrDict[Register, Register] = AttrDict()
        # 可以使用的结果寄存器
        result_regs = len(self.regs.valid_result_regs)

        # region: 约束编程模型
        model = cp_model.CpModel()
        max_value = max(max(reg.used_by) for reg in regs_to_merge)

        reg_ids = []  # 放到第几个空余寄存器
        usages = []
        for i, reg in enumerate(regs_to_merge):
            reg_ids += [model.NewIntVar(0, result_regs - 1, f"real_reg_index_for_reg_{i}")]
            usages += [
                model.NewFixedSizeIntervalVar(
                    start=(reg_ids[-1] * max_value) + reg.first + 1,
                    size=reg.last - reg.first,
                    name=f"usage_{i}",
                )
            ]

        model.AddNoOverlap(usages)
        model.Minimize(sum(reg_ids))
        # endregion: 约束编程模型

        # region: 约束编程求解
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        for status_str in ["unknown", "model_invalid", "feasible", "infeasible", "optimal"]:
            if status == getattr(cp_model, status_str.upper()):
                status_str = status_str.upper()
                break

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            err_msg = [f"求解失败: {status_str}."]
            err_msg += [f"{self.smt_info_str}"]
            err_msg += ["函数输入:"]
            for line in self.parsed_ir["func_body"]["statements"]:
                err_msg += [str(line)]
            raise RuntimeError("\n".join(err_msg))
        # endregion: 约束编程求解

        for old_reg_id, real_reg_id in enumerate(reg_ids):
            reg_id = solver.Value(real_reg_id)
            result[regs_to_merge[old_reg_id]] = self.regs.valid_result_regs[reg_id]
            result[regs_to_merge[old_reg_id]].used_by |= regs_to_merge[old_reg_id].used_by
        return result

    def try_use_result_reg(self) -> None:
        """合并虚拟寄存器到 `ADD_S` 或者 `MUL_P`.

        只有马上使用到的且以后不会再使用的结果会使用 `ADD_S` 或者 `MUL_P`.

        更新 `self._reg_results`.
        """

        not_nop_stmts: list[tuple[int, SMT]] = []
        for ir_stmt_id, smt_stmts in self._smt_results.items():
            smt_stmts: list[SMT]
            for smt_stmt in smt_stmts:
                if smt_stmt.op == Operator.NOP:
                    continue
                not_nop_stmts += [(ir_stmt_id, smt_stmt)]

        for i, (ir_stmt_id, smt_stmt) in enumerate(not_nop_stmts):
            if i == len(not_nop_stmts) - 1:
                break

            ir_stmt_id: int
            smt_stmt: SMT

            if smt_stmt.operator == "add":
                result_reg = smt_stmt.s
                new_reg = self.regs.ADD_S
            elif smt_stmt.operator == "mul":
                result_reg = smt_stmt.p
                new_reg = self.regs.MUL_P
            else:
                continue

            if not result_reg.name.startswith("DUMMY_"):
                continue

            if len(result_reg.used_by) > 2:
                continue

            next_stmt = not_nop_stmts[i + 1][1]

            if result_reg not in next_stmt.input_regs:
                continue

            multiple_usage = False
            for _, stmt in not_nop_stmts[i + 2 :]:
                if result_reg in stmt.input_regs:
                    multiple_usage = True
                    break

            if multiple_usage > 1:
                continue

            # print(f"使用 {new_reg} 代替 {result_reg}.")
            next_stmt.update_regs(old_reg=result_reg, new_reg=new_reg)

            result_reg.release()

            if smt_stmt.operator == "add":
                smt_stmt.s = self.regs.ADD_S
            else:
                smt_stmt.p = self.regs.MUL_P

    _final_smt_result: List[SMT] = None
    """SMT 语句
    """

    def get_smt_result(self) -> List[SMT]:
        """返回得到 SMT 语句"""

        if self._final_smt_result:
            return self._final_smt_result

        self.compile()

        smt_info = []
        smt_info += [""]
        smt_info += ["未合并的虚拟寄存器"]
        for reg in self.regs.all_registers.values():
            if reg.index < 32:
                continue
            if reg.used_by:
                smt_info += [str(reg)]
        smt_info += [""]
        smt_info += ["未优化的 SMT 语句"]
        for stmts in self._smt_results.values():
            for stmt in stmts:
                if stmt.op == Operator.NOP:
                    continue
                smt_info += [str(stmt)]

        self.smt_info_str = "\n".join(smt_info)
        reg_map = self.optimize()
        result: List[SMT] = reduce(lambda a, b: a + b, self._smt_results.values())

        # NOTE, uger, 并删除一条间隔的NOP指令
        for smt_idx, smt_cmd in enumerate(result):
            smt_cmd: SMT
            smt_cmd.result_bits = self.regs.result_bits

            if smt_cmd.op != Operator.CALCU:
                continue

            for arg_name in ["a1", "a2", "s", "m1", "m2", "p"]:
                if (arg_reg := getattr(smt_cmd, arg_name)) not in reg_map:
                    continue
                setattr(smt_cmd, arg_name, reg_map[arg_reg])

            # 使用 3'b111 或者 4'b1111 作为计算结果表示输出到 ADD_S 或 MUL_P
            if smt_cmd.reg_result in [self.regs.ADD_S, self.regs.MUL_P]:
                smt_cmd.reg_result = self.regs.NONE_REG

                result.pop(smt_idx+1)

        self.used_shared_regs = self.regs.used_shared_reg
        self._final_smt_result = result
        return self._final_smt_result

    @classmethod
    def compile_all(
        cls,
        func: Dict[str, Callable],
        preload_constants: Dict[str, Number] = None,
        predefined_regs: Dict[str, str] = None,
        i_func: Callable = None,
        update_method: Dict[str, str] = None,
        result_bits: int = None,
    ) -> tuple[SMTCompiler, SMTCompiler, list[SMT]]:
        """编译 I 函数和其他函数.

        Args:
            func (dict[str, Callable]): 函数, 比如 {"V": xxx, "u": xxx}
            preload_constants (dict[str, Number]): 预先载入常数.
            i_func (Callable, Optional): I 函数.
            update_method (dict[str, str], Optional): 参数的更新方法默认累加, e.g. {"V": "acc", "g1": "update", "g2": "update"}
            result_bits (int, Optional): 结果寄存器位宽. 3 或者 4.

        Returns:
            tuple[SMTCompiler, SMTCompiler, list[SMT]]: I 函数编译器,
                其他编译器, SMT 语句.
        """
        constants = AttrDict(preload_constants)

        if result_bits is None:
            v_compiler = cls(func=func, is_i_func=False, update_method=update_method)
        else:
            v_compiler = cls(func=func, is_i_func=False, update_method=update_method, result_bits=result_bits)

        compilers = [v_compiler]
        if i_func:
            if result_bits is None:
                i_compiler = cls(func={"I": i_func}, is_i_func=True, update_method=update_method)
            else:
                i_compiler = cls(
                    func={"I": i_func}, is_i_func=True, update_method=update_method, result_bits=result_bits
                )
            compilers = [i_compiler] + compilers
        else:
            i_compiler = None

        smt_result = []
        i_reg_name = ""
        used_arg_names =  predefined_regs
        used_shared_regs = set()
        used_regs = set()
        for compiler in compilers:
            regs = compiler.smt_factory.regs
            pc = compiler.preload_constants
            for reg_name in used_regs:
                if reg_name in ["R0", "R1"]:  # R0 R1 不会作为函数输入所以不用跳过
                    continue
                reg = regs.get_reg_by_name(reg_name)
                reg.used_by.add(-3)  # 被上一批函数使用, e.g. 计算 I

            compiler.used_arg_names = used_arg_names

            # # 预定义的寄存器
            # for reg_alias, reg_name in predefined_regs.items():
            #     reg = regs.get_reg_by_name(reg_name)
            #     reg.update(alias=reg_alias, used_by={-2})

            # 预先载入常数
            if not used_shared_regs:
                for c_reg in pc:
                    if c_reg.alias == "V_reset":
                        c_reg.value = constants.V_reset
                    elif c_reg.alias == "T_refrac":
                        c_reg.value = constants.tau_ref

                reg = compiler.regs.unused_shared_reg.update(alias="V_thresh", value=constants.V_th, used_by={-2})
                pc.add(reg)
                compiler.V_thresh = reg

                reg = compiler.regs.unused_shared_reg.update(alias="V_rest", value=constants.V_rest, used_by={-2})
                pc.add(reg)
                # compiler.V_reset = reg
            else:
                compiler.i_reg_name = i_reg_name
                compiler.used_arg_names = used_arg_names
                compiler.used_shared_regs = used_shared_regs
                for old_reg in compiler.used_shared_regs:
                    new_reg = compiler.regs.all_registers[old_reg.index]
                    new_reg.used_by.add(-2)
                    new_reg.update(value=old_reg.value, alias=old_reg.alias)
                    pc.add(new_reg)
                    if new_reg.alias == "V_thresh":
                        compiler.V_thresh = new_reg
                    elif new_reg.alias == "V_reset":
                        compiler.V_reset = new_reg

            compiler.compile()
            smt_result += compiler.get_smt_result()

            for reg in compiler.regs.all_registers.values():
                if reg.used_by:
                    used_regs.add(reg.name)

            if compiler.is_i_func:
                i_reg_name = compiler.i_reg_name
                used_shared_regs = compiler.used_shared_regs
                used_arg_names = compiler.used_arg_names
        sram_load = v_compiler.smt_factory.sram_load()
        sram_load[0].result_bits = result_bits
        smt_result = sram_load + smt_result
        return (i_compiler, v_compiler, smt_result)
