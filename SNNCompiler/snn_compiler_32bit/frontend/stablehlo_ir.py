"""Stablehlo IR 语句
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Dict

import jax
from addict import Dict as AttrDict
from brainpy._src.integrators import JointEq
from strenum import StrEnum

from ..common.register import Register
from ..frontend.stablehlo_parser import stablehlo_parser


class IRCmd(StrEnum):
    """支持的 IR 命令类型"""

    negate = "negate"
    """取负值

    - 常数, 正值不被直接使用, 负值被使用: 存储负值到共享寄存器
    - 常数, 正值和负值都被使用: 存储正值和负值到两个共享寄存器
    - 运算结果, 结果不被直接使用, 负值被使用: 存储运算结果到 `smt.R6` 或 `smt.R5`,
        存储 `smt.R6_NEG`+ `ZERO_REG` 到共享寄存器
    - 运算结果, 结果和负值都被使用: 存储运算结果到 `smt.R6` 或 `smt.R5`,
        存储 (`smt.R6` 或 `smt.R5`) + `ZERO_REG` 到共享寄存器
        存储 (`smt.R6_NEG` 或 `smt.R5_NEG`) + `ZERO_REG` 到另一个共享寄存器
    - 优先使用 `smt.R6`, 如果被 V 占用则使用 `smt.R5`.
    """

    constant = "constant"
    """常数

    直接保存到共享寄存器
    """

    add = "add"
    """加法

    直接输出和到共享寄存器
    """

    multiply = "multiply"
    """乘法

    直接输出乘积到共享寄存器
    """

    divide = "divide"
    """除法

    - 除数为常数, 只被除法 `IRCmd.divide`: 存储除数的倒数到共享寄存器
        - 优化方向: 类型转换 `IRCmd.convert`, 取负值 `IRCmd.negate` 不占用寄存器
    - 除数为常数, 被其他语句使用: 存储常数和除数的倒数到共享寄存器
        - 优化方向: 类型转换 `IRCmd.convert`, 取负值 `IRCmd.negate` 不占用寄存器
    - 除数为运算结果: 不支持

    之后运行乘法并输出乘积到共享寄存器
    """

    power = "power"
    """指数, 暂时只支持非负整数常数.

    直接输出乘积到共享寄存器.
    """

    convert = "convert"
    """类型转换

    直接输出到共享寄存器
    """


@dataclass
class IRStatement:
    """IR 语句, 操作数可以是寄存器对象."""

    reg_index: int
    """运算结果寄存器索引
    """

    cmd: IRCmd
    """命令类型
    """

    operands: list[Register | int]
    """操作数

    - 寄存器对象: 函数输入或者常数
    - 结果寄存器索引: 运算结果
    """


class ParsedIR(AttrDict):
    """解析过的 IR.

    - `module_head`: 函数信息.
    - `func_head`: 函数头.
    - `func_body`: 函数体.
    """

    func_group: FuncGroup
    """函数组对象
    """

    @classmethod
    def load_one_func(cls, func: Callable) -> ParsedIR:
        """从函数得到的解析过的 stablehlo IR.

        Args:
            func (Callable): 函数

        Returns:
            ParsedIR: 解析过的 stablehlo IR.
        """
        func_sig = inspect.signature(func)
        arg_names = str(func_sig).strip("()").split(",")
        lower_arg = list(range(len(arg_names)))
        # NOTE, 保留keep_unused 
        func_jit = jax.jit(func, keep_unused=True)
        func_ir = str(func_jit.lower(*lower_arg).compiler_ir(dialect="stablehlo"))
        result = ParsedIR(stablehlo_parser.parse_string(func_ir).as_dict())
        del result.func_body.return_statement["dtype"]
        
        # region: jax>=0.4.14 取消了 jax.arg_info 属性, 添加
        for arg_def, arg_name in zip(result.func_head.arg_defs, arg_names):
            for info in arg_def.info:
                if info.name == ["jax", "arg_info"]:
                    if info.value.startswith("I"):
                        info.value = "I"  # 所有以 I 开头的参数都强制改名为 I
                    break  # jax.arg_info 已经存在
            else:
                arg_name = arg_name.strip()
                if arg_name.startswith("I"):  # 所有以 I 开头的参数都强制改名为 I
                    arg_name = "I"
                arg_def.info.append(AttrDict(name=["jax", "arg_info"], value=arg_name))
        # endregion
        
        return result

    @classmethod
    # pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
    def load(cls, func: Dict[str, Callable]) -> ParsedIR:
        """从函数得到的解析过的 stablehlo IR.

        Args:
            func (dict[str, Callable]): 函数返回变量名称 -> 函数.

        Returns:
            ParsedIR: 解析过的 stablehlo IR.
        """

        result = ParsedIR()
        result.func_group = Func.load(func)
        module_head = None

        func_head = AttrDict()
        func_head.name = "@main"
        func_head.arg_defs = []
        func_head.return_def = AttrDict()

        func_body = AttrDict()
        func_body.statements = []

        known_args = set()
        reg_index_base = 0

        for one_func in result.func_group:
            one_func: Func
            parsed_ir = ParsedIR.load_one_func(one_func.body)
            module_head = module_head or parsed_ir.module_head

            old_args = parsed_ir.func_arg_names

            for one_arg_def in parsed_ir.func_head.arg_defs:
                add_arg = ""
                for one_info in one_arg_def.info:
                    if one_info.name == ["jax", "arg_info"]:
                        add_arg = one_info.value
                        break
                if not add_arg:
                    continue
                if add_arg in known_args:
                    continue
                known_args.add(add_arg)
                arg_def = AttrDict()
                arg_def.index = len(func_head.arg_defs)
                arg_def.dtype = one_arg_def.dtype
                arg_def.info = one_arg_def.info
                func_head.arg_defs.append(arg_def)

            for stmt in parsed_ir.func_body.statements:
                stmt.reg_index = len(func_body.statements)
                for opr in stmt.operands:
                    if opr.type == "reg_index":
                        opr.value += reg_index_base
                    elif opr.type == "arg_index":
                        opr.value = old_args[opr.value]
                func_body.statements.append(stmt)

            reg_index_base = len(func_body.statements)

            if func_body.return_statement:
                func_body.return_statement.return_dtypes += parsed_ir.func_body.return_statement.return_dtypes
                parsed_ir.func_body.return_statement.operands[0].value = func_body.statements[-1].reg_index
                func_body.return_statement.operands += parsed_ir.func_body.return_statement.operands
            else:
                func_body.return_statement = parsed_ir.func_body.return_statement

            func_body.return_statement.operands[-1].name = one_func.returns

        module_head.at_identifier = "@jit_" + ("_".join(result.func_group.returns))
        result.module_head = module_head
        result.func_head = func_head
        result.func_body = func_body
        if "dtype" in result.func_body.return_statement:
            del result.func_body.return_statement["dtype"]

        return result

    @property
    def func_arg_names(self) -> AttrDict[int, str]:
        """输入函数的参数名称.

        Returns:
            AttrDict[int, str]: 函数参数, e.g. `{0: "V", 1: "I"}`
        """
        result = AttrDict()
        for arg in self.func_head.arg_defs:
            for info in arg.info:
                if "arg_info" in info.name:
                    name = info.value
                    break
            else:
                raise RuntimeError(f"函数参数没有名字: {arg}")
            result[arg.index] = name
        return result


@dataclass
class Func:
    """一个函数, 包括:

    - `args`: 函数参数变量名字
    - `body`: 函数对象
    - `results`: 函数结果变量名字
    """

    args: list[str]
    """函数参数
    """

    body: Callable
    """函数算式
    """

    returns: str
    """函数结果变量名字
    """

    @classmethod
    def load(cls, func: Dict[str, Callable]) -> FuncGroup:
        """读取函数或函数组为 `Func` 对象.

        Args:
            func (dict[str, Callable]): 函数返回值名称 -> 函数或函数组 (`JointEq`).
                如果 func 的值是函数组 (`JointEq`) 则忽略返回值名称.

        Returns:
            FuncGroup: 函数组对象
        """
        result = FuncGroup()
        for return_name, one_func in func.items():
            if isinstance(one_func, JointEq):  # 读取函数组信息
                # 虽然可以通过函数名称推导返回值名称但是怕 brainpy 内部结构改变
                for i, result_vars in enumerate(one_func.vars_in_eqs):
                    result += [Func(args=one_func.args_in_eqs[i], body=one_func.eqs[i], returns=result_vars[0])]
            else:
                parsed_ir = ParsedIR.load_one_func(one_func)
                result += [Func(args=list(parsed_ir.func_arg_names.values()), body=one_func, returns=return_name)]
        return result


class FuncGroup(list):
    """函数组, 对象属性:

    - `args`: 所有函数的参数变量名字
    - `results`: 所有函数的结果变量名字
    """

    @property
    def args(self) -> list[str]:
        """所有函数的参数.

        Returns:
            list[str]: 所有函数的参数.
        """
        result = []
        for one_func in self:
            if not isinstance(one_func, Func):
                raise RuntimeError(f"{one_func} is NOT Func but {type(one_func)}")
            for one_arg in one_func.args:
                if one_arg not in result:
                    result += [one_arg]
        return result

    @property
    def returns(self) -> list[str]:
        """所有函数的结果变量名.

        Returns:
            list[str]: 所有函数的结果变量名.
        """
        result = []
        for one_func in self:
            if not isinstance(one_func, Func):
                raise RuntimeError(f"{one_func} is NOT Func but {type(one_func)}")

            if one_func.returns not in result:
                result += [one_func.returns]
        return result
