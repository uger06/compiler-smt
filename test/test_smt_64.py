"""测试 `SMTCompiler`
"""

import math
import random
from typing import Callable

import brainpy as bp
import pytest
from addict import Dict as AttrDict
from loguru import logger

from SNNCompiler.snn_compiler_64bit.flow.smt_64_compiler import SMT64Compiler

random.seed(42)

# region: constant values 常数数值
cv = AttrDict()  # constant values 常数数值
cv.R = random.randint(0, 99)
cv.tau = random.randint(0, 99)
cv.tau_ref = random.randint(0, 99)
cv.delay = random.randint(0, 99)
cv.c = random.randint(0, 99)
cv.v_c = random.randint(0, 99)
cv.a = random.randint(0, 99)
cv.b = random.randint(0, 99)
cv.gNa = random.randint(0, 99)
cv.ENa = random.randint(0, 99)
cv.gK = random.randint(0, 99)
cv.EK = random.randint(0, 99)
cv.gL = random.randint(0, 99)
cv.EL = random.randint(0, 99)
cv.v_rest = random.randint(-100, 0)
cv.v_thresh = random.randint(-100, 0)
cv.v_reset = random.randint(-100, 0)
cv.t_refrac = random.randint(0, 99)
cv.E1 = random.randint(0, 99)
cv.E2 = random.randint(0, 99)
# endregion: constant values 常数数值


def hh_0(V, m, h, n, I):
    """Hindmarsh-Rose model"""
    I_Na = (cv.gNa * m**3.0 * h) * (V - cv.ENa)
    I_K = (cv.gK * n**4.0) * (V - cv.EK)
    I_leak = cv.gL * (V - cv.EL)
    return (-I_Na - I_K - I_leak + I) / cv.c


def hh_1(V, m, h, n, I):
    """Hindmarsh-Rose model, flattened"""
    return (
        -(cv.gNa * m * m * m * h) * (V - cv.ENa) - (cv.gK * n * n * n * n) * (V - cv.EK) - (cv.gL * (V - cv.EL)) + I
    ) / cv.c


func_examples = {
    "0": lambda V, I: (cv.v_rest * V / 3) - (cv.v_thresh + cv.v_reset) - I,
    "1": lambda V, I: (-V + cv.v_rest + I) / 20,
    "2": lambda V, I: (-V + cv.v_rest - I) / 20,
    "3": lambda V, I: (-V * V * V + cv.v_rest + cv.R * I) / cv.tau,
    "4": lambda V, I: (-(V - cv.v_rest) * V * V + cv.v_reset + cv.R * I) / cv.tau,
    "5": lambda V, I, u: (-V * V * V + cv.v_rest - cv.delay * u + cv.R * I) / cv.tau,
    "6": lambda V, I: (cv.c * (V - cv.v_rest) * (V - cv.v_c) + cv.R * I) / cv.tau,
    "7": lambda V, I, u: 0.04 * V * V + 5 * V + 140 - u + I,
    "8": lambda V, I, t, y, z: y - cv.a * V * V * V + cv.b * V * V - z + I,
    "hh_0": hh_0,
    "hh_1": hh_1,
    "brainpy_lif": bp.neurons.LIF(
        256,
        V_rest=cv.v_rest,
        V_th=cv.v_thresh,
        V_reset=cv.v_reset,
        tau=cv.tau,
        tau_ref=cv.tau_ref,
        method="exp_auto",
        V_initializer=bp.init.Normal(-55.0, 2.0),
    ).derivative,
    "QuaIF": bp.neurons.QuaIF(256).derivative,
    "Izhikevich": bp.dyn.Izhikevich(4000).derivative,
    "HindmarshRose": bp.neurons.HindmarshRose(256).derivative,
    "ExpIF": bp.neurons.ExpIF(256).derivative,
    "AdExIF": bp.neurons.AdExIF(256).derivative,
    "HH": bp.neurons.HH(256).derivative,
}


class EINet(bp.DynSysGroup):  # pylint: disable=missing-class-docstring
    def __init__(self):
        super().__init__()
        ne, ni = 3200, 800
        self.E = bp.dyn.LifRef(
            ne, V_rest=-60.0, V_th=-50.0, V_reset=-60.0, tau=20.0, tau_ref=5.0, V_initializer=bp.init.Normal(-55.0, 2.0)
        )
        self.I = bp.dyn.LifRef(
            ni, V_rest=-60.0, V_th=-50.0, V_reset=-60.0, tau=20.0, tau_ref=5.0, V_initializer=bp.init.Normal(-55.0, 2.0)
        )
        self.E2E = bp.dyn.ProjAlignPostMg2(
            pre=self.E,
            delay=0.1,
            comm=bp.dnn.EventJitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
            syn=bp.dyn.Expon.desc(size=ne, tau=5.0),
            out=bp.dyn.COBA.desc(E=0.0),
            post=self.E,
        )
        self.E2I = bp.dyn.ProjAlignPostMg2(
            pre=self.E,
            delay=0.1,
            comm=bp.dnn.EventJitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
            syn=bp.dyn.Expon.desc(size=ni, tau=5.0),
            out=bp.dyn.COBA.desc(E=0.0),
            post=self.I,
        )
        self.I2E = bp.dyn.ProjAlignPostMg2(
            pre=self.I,
            delay=0.1,
            comm=bp.dnn.EventJitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
            syn=bp.dyn.Expon.desc(size=ne, tau=10.0),
            out=bp.dyn.COBA.desc(E=-80.0),
            post=self.E,
        )
        self.I2I = bp.dyn.ProjAlignPostMg2(
            pre=self.I,
            delay=0.1,
            comm=bp.dnn.EventJitFPHomoLinear(ni, ni, prob=0.02, weight=6.7),
            syn=bp.dyn.Expon.desc(size=ni, tau=10.0),
            out=bp.dyn.COBA.desc(E=-80.0),
            post=self.I,
        )


@pytest.mark.parametrize("name,func", func_examples.items())
def test_smt_96_compiler(name: str, func: Callable) -> None:
    """测试 `SMT96Compiler`"""

    model = EINet()
    index = 0
    for proj in model.nodes().subset(bp.Projection).values():
        if proj.refs["post"] != model.I:
            continue
        index += 1
        cv[f"E{index}"] = proj.refs["out"].E
        cv[f"tau{index}"] = proj.refs["syn"].tau

    i_func = {
        "I": lambda g1, g2, V: cv.R * g1 * (cv.E1 - V) + cv.R * g2 * (cv.E2 - V),
    }

    g_func = {
        "g1": lambda g1: g1 * math.exp(-1 / cv.tau1),
        "g2": lambda g2: g2 * math.exp(-1 / cv.tau2),
    }

    # funcs = {"V": func}
    # if name in ["HindmarshRose"]:
    #     _, _, smt_result = SMT96Compiler.compile_all(funcs=funcs)
    # else:
    #     funcs.update(g_func)
    #     funcs.update(i_func)
    #     _, _, smt_result = SMT96Compiler.compile_all(
    #         funcs=funcs,
    #         update_method={"I": "update", "g1": "update", "g2": "update", "V": "acc"},
    #         predefined_reg={"V": "R3", "g1": "R4", "g2": "R5", "I": "R6"},
    #     )
    # logger.critical(smt_result)


    funcs = {"V": func}
    funcs.update(g_func)
    funcs.update(i_func)
    
    _, v_compiler, smt_result = SMT64Compiler.compile_all(
    funcs=funcs,
    predefined_regs={"V": "R3", "g1": "R4", "g2": "R5", "I": "R6"},
    )
    logger.critical(smt_result)
