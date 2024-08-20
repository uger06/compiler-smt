from SNNCompiler.snn_compiler_64bit.backend.smt_64_op import CalField, OPType, RD_OPType, RS_OPType, ALU_OPType
from SNNCompiler.snn_compiler_64bit.backend.smt_64_stmt.smt64 import SMT64

from SNNCompiler.snn_compiler_64bit.common.smt_64_reg import  RegisterCollection
from SNNCompiler.snn_compiler_64bit.backend.smt_64_stmt.cal_op import CAL_OP 
from SNNCompiler.snn_compiler_64bit.backend.smt_64_stmt.nop import NOP 
from SNNCompiler.snn_compiler_64bit.flow.smt_64_compiler import SMT64Compiler

import brainpy as bp
# funcs = {"V": bp.neurons.LIF(256).derivative}
# compiler = SMT64Compiler(funcs=funcs)
# compiler.update_stablehlo_statements()

funcs = {"V": bp.neurons.LIF(
            256,
            V_rest=-52,
            V_th=-67,
            V_reset=-22,
            tau=31,
            tau_ref=77,
            method="exp_auto",
            V_initializer=bp.init.Normal(-55.0, 2.0),).derivative}
"""
\tau \frac{dV}{dt} = - (V(t) - V_{rest}) + RI(t)
"""


funcs = {"V": bp.neurons.QuaIF(
            256,
            V_rest=-66,
            V_th=-67,
            V_reset=-22,
            tau=31,
            method="exp_auto",
            V_initializer=bp.init.Normal(-55.0, 2.0),).derivative}
"""\tau \frac{d V}{d t}=c(V-V_{rest})(V-V_c) + RI(t)"""


funcs = {"V": bp.neurons.Izhikevich(
            256).derivative}
"""
    \frac{d V}{d t} &= 0.04 V^{2}+5 V+140-u+I
    \frac{d u}{d t} &=a(b V-u)
"""


funcs = {"V": bp.neurons.ExpIF(
            256).derivative}
"""
\tau\frac{d V}{d t}= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} + RI(t), \\

"""

funcs = {"V": bp.neurons.AdExIF(
            256).derivative}
"""
\tau_m\frac{d V}{d t} &= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} - Rw + RI(t), \\
\tau_w \frac{d w}{d t} &=a(V-V_{rest}) - w

"""


funcs = {"V": bp.neurons.HH(
            256).derivative}
"""
      C \frac {dV} {dt} = -(\bar{g}_{Na} m^3 h (V &-E_{Na})
      + \bar{g}_K n^4 (V-E_K) + g_{leak} (V - E_{leak})) + I(t)

      \frac {dx} {dt} &= \alpha_x (1-x)  - \beta_x, \quad x\in {\rm{\{m, h, n\}}}

      &\alpha_m(V) = \frac {0.1(V+40)}{1-\exp(\frac{-(V + 40)} {10})}

      &\beta_m(V) = 4.0 \exp(\frac{-(V + 65)} {18})

      &\alpha_h(V) = 0.07 \exp(\frac{-(V+65)}{20})

      &\beta_h(V) = \frac 1 {1 + \exp(\frac{-(V + 35)} {10})}

      &\alpha_n(V) = \frac {0.01(V+55)}{1-\exp(-(V+55)/10)}

      &\beta_n(V) = 0.125 \exp(\frac{-(V + 65)} {80})
"""


constants = {"V_th": 2.0, # 硬件ASIC输入的常数
            "I_x": 3.0}

predefined_regs = {
    "V": "R3",
    "I": "R6",
}


_, v_compiler, statements = SMT64Compiler.compile_all(funcs=funcs, 
                                                    #   constants=constants,
                                                      predefined_regs=predefined_regs)



flag = 1