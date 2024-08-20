"""stablehlo to SMT
"""
from functools import cached_property
from typing import Dict, List

from addict import Dict as AttrDict

from ..backend.smt import Register
from ..frontend.stablehlo_ir import IRStatement, ParsedIR
from .smt_compiler_base import SMTCompilerBase


class SMTCompilerBaseWithMethods(SMTCompilerBase):
    """SMT 编译器的基础方法."""

    @cached_property
    def parsed_ir(self) -> ParsedIR:
        """从函数得到的解析过的 stablehlo IR.

        如果输入是函数组`JointEq`, 则合并所有函数体.

        Returns:
            ParsedIR: 解析过的 stablehlo IR.
        """
        result = ParsedIR.load(self.func)
        for i, return_name in enumerate(self.func):
            result.func_body.return_statement.operands[i].name = return_name
        return result

    @cached_property
    def func_args(self) -> Dict[int, Register]:
        """输入函数的参数.

        Returns:
            AttrDict[int, Register]: 函数参数, e.g. `{0: V, 1: I}`,
        """
        result: dict[int, Register] = AttrDict()
        arg_names = []
        for i, (name, reg_name) in enumerate(self.used_arg_names.items()):
            reg = self.regs.get_reg_by_name(reg_name)
            reg.update(alias=name, used_by={-1}, as_arg=name)
            result[i] = reg
            arg_names.append(name)

        for i, name in self.parsed_ir.func_arg_names.items():
            if name in arg_names:
                continue
            if (not self.is_i_func) and name.startswith("I") and self.i_reg_name:
                # I 运算
                reg = self.regs.get_reg_by_name(name=self.i_reg_name)
            elif name == "V":
                reg = self.regs.V
            else:
                # 其他输入
                if self.update_method.get(name, None) or (name in self.return_names.values()):
                    # 需要存在结果寄存器, 不然会多需要一个寄存器
                    for reg in reversed(self.regs.valid_func_arg_regs):
                        if reg.used_by:
                            continue
                        if reg not in self.regs.valid_result_regs:
                            continue
                        self.regs.use_reg(reg)
                        break
                    else:
                        raise RuntimeError("找不到未被占用的可作为结果寄存器的函数参数寄存器")
                else:
                    reg = self.regs.unused_arg_reg

            if name.startswith("I"):
                # 所有以 I 开头的参数都是 I
                name = "I"

            reg.update(alias=name, used_by={-1}, as_arg=name)
            result[len(result)] = reg

        for reg in result.values():
            if reg in self.regs.valid_result_regs:
                self.regs.valid_result_regs.remove(reg)
            if reg in self.regs.valid_func_arg_regs:
                self.regs.valid_func_arg_regs.remove(reg)
        return result

    @cached_property
    def return_names(self) -> Dict[int, str]:
        """返回所有输出变量的名字, e.g. {8: "V"} 表示第 8 条语句的结果为 V 输出.
        注意: 运算阶段不要更新.

        Returns:
            dict[int, str]: 所有输出语句编号和对应变量名称.
        """
        result: dict[int, str] = AttrDict()
        for opr in self.parsed_ir.func_body.return_statement.operands:
            result[opr.value] = opr.name
        return result

    def get_result_reg(self, stmt_id: int) -> Register:
        """返回没占用的共享寄存器作为结果寄存器并记录在 `self._reg_results`.
        `result.used_by = {stmt_id}`.

        Returns:
            Register: 结果寄存器.
        """
        if self._reg_results[stmt_id]:
            raise RuntimeError(f"结果寄存器 {stmt_id} 已经被占用.")

        result = self.regs.unused_dummy_reg

        if stmt_id in self.return_names:
            result.as_return = self.return_names[stmt_id]
            return_index = next(i for i, v in enumerate(self.return_names.values()) if v == result.as_return)
            result.used_by.add(return_index + len(self.parsed_ir.func_body.statements))

        result.used_by.add(stmt_id)
        self._reg_results[stmt_id] = result
        return result

    def update_result_reg(self, stmt_id: int, reg: Register) -> None:
        """使用 `reg` 更新 `self._reg_results` 以及 `ir_stmts` 操作数寄存器.

        Args:
            stmt_id (int): IR 语句索引.
            new_reg (Register): 新的寄存器.
        """
        old_reg: Register = self._reg_results[stmt_id]
        replaced = False
        for ir_stmt in self.ir_stmts:
            if old_reg not in ir_stmt.operands:
                continue
            for i, r in enumerate(ir_stmt.operands):
                if old_reg == r:
                    ir_stmt.operands[i] = reg
                    replaced = True
        if replaced:
            reg.used_by = old_reg.used_by | reg.used_by
            old_reg.release()
        self._reg_results[stmt_id] = reg

    @cached_property
    def ir_stmts(self) -> List[IRStatement]:
        """IR 指令列表. 操作数为:

        - 寄存器对象: 函数输入或者常数
        - 结果寄存器索引: 运算结果

        Returns:
            list[IRStatement]: IR 指令列表.
        """
        result: List[IRStatement] = []
        for stmt_id, stmt in enumerate(self.parsed_ir.func_body.statements):
            ir_stmt = IRStatement(reg_index=stmt_id, cmd=stmt.cmd, operands=[])
            self.get_result_reg(stmt_id=stmt_id)  # 初始化结果寄存器占位符
            for opr in stmt.operands:
                if opr.type == "arg_index":  # 函数输入
                    if opr.value.startswith("I"):
                        opr.value = "I"
                    reg = next(r for r in self.func_args.values() if r.as_arg == opr.value)
                elif opr.type == "constant":  # 常数
                    reg = self.add_constant_reg(opr.value)
                elif opr.type == "reg_index":  # 之前的计算结果
                    reg = self._reg_results[opr.value]  # IR 语句索引 == 结果寄存器索引
                else:
                    raise NotImplementedError(f"暂不支持 {opr.type = }, {opr.value = }")
                reg.used_by.add(stmt_id)
                ir_stmt.operands += [reg]
            result += [ir_stmt]
        return result
