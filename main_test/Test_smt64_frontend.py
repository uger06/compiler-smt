import inspect
import jax
from SNNCompiler.snn_compiler_64bit.frontend.stablehlo_parser import StableHLOProgram
import brainpy as bp
from SNNCompiler.snn_compiler_96bit.backend.smt_96_stmt.fp_op import FP_OP
from SNNCompiler.snn_compiler_96bit.common.smt_96_base import CTRL_LEVEL, CTRL_PULSE, IBinary, IEEE754

from SNNCompiler.snn_compiler_96bit.backend.smt_96_factory import SMT96Factory
from SNNCompiler.snn_compiler_96bit.backend.smt_96_stmt import NOP, SMT96
from SNNCompiler.snn_compiler_96bit.common.smt_96_reg import Register96, RegisterCollection
from SNNCompiler.snn_compiler_64bit.flow.smt_64_compiler import SMT64Compiler

import brainpy.math as bm

# def func_for_ir_exp(V, V_rest):
#     '''测试函数'''
#     return 42 * bm.exp(V/V_rest - V_rest) + V_rest

# def func_for_ir(V, V_rest):
#     '''测试函数'''
#     return 42 * (V * V - V_rest) + V_rest

# import brainpy as bp
# funcs = {"V": func_for_ir_exp}
# compiler = SMT64Compiler(funcs=funcs)
# compiler.update_stablehlo_statements()

       
def func_for_ir(V, V_rest):
    '''测试函数'''
    return 42 * (V/V_rest - V_rest) + V_rest

def func_for_ir_exp(V, V_rest):
    '''测试函数'''
    return 42 * bm.exp(V/V_rest - V_rest) + V_rest

def func_for_ir_single(V):
    '''测试函数'''
    return 52 - bm.log(-V) + bm.exp(-10.0) + 20.0

# funcs = {"V": bp.neurons.LIF(256).derivative}


rog = StableHLOProgram.load_one_func(func_for_ir_single)

# funcs = {"HindmarshRose": bp.neurons.HindmarshRose(256).derivative}
funcs = {"ExpIF": bp.neurons.ExpIF(256).derivative}
prog = StableHLOProgram.load(funcs)

flag = 1 
