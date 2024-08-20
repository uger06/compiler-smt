from SNNCompiler.func2smt_64bit import SMT64bit
import brainpy as bp
import numpy as np
import jax 
import random 
import warnings
from brainpy import math as bm

warnings.filterwarnings('ignore')

class Exponential(bp.Projection): 
  def __init__(self, pre, post, delay, prob, g_max, tau, E):
    super().__init__()
    self.pron = bp.dyn.FullProjAlignPost(
      pre=pre,
      delay=delay,
      # comm=bp.dnn.EventJitFPHomoLinear(pre.num, post.num,prob=prob, weight=g_max, seed = 42),             
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num,seed = 42), g_max), 
      syn=bp.dyn.Expon(size=post.num, tau=tau),# Exponential synapse
      out=bp.dyn.COBA(E=E), # COBA network
      post=post
    )


class EINet(bp.DynamicalSystem):
  def __init__(self, ne=3200, ni=800):
    super().__init__()
    self.E = bp.dyn.LifRef(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                           )
    self.I = bp.dyn.LifRef(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                           )
    
    self.E2E = Exponential(self.E, self.E, delay=0., prob=0.02, g_max=0.6, tau=5., E=0.)
    self.E2I = Exponential(self.E, self.I, delay=0., prob=0.02, g_max=0.6, tau=5., E=0.)
    self.I2E = Exponential(self.I, self.E, delay=0., prob=0.02, g_max=6.7, tau=10., E=-80.)
    self.I2I = Exponential(self.I, self.I, delay=0., prob=0.02, g_max=6.7, tau=10., E=-80.)

  def update(self, inp=0.):
    self.E2E()
    self.E2I()
    self.I2E()
    self.I2I()
    self.E(inp)
    self.I(inp)
    # monitor
    return self.E.spike, self.I.spike
  
  def run(self,T):
    E_sps = []
    I_sps = []
    for i in range(T):
        bp.share.save(t=i)
        self.update(20.)
        E_sps.append(self.E.spike.value)
        I_sps.append(self.I.spike.value)
    E_sps = np.array(E_sps)
    I_sps = np.array(I_sps)
    return E_sps,I_sps



if __name__ == "__main__":
    
    bm.set_dt(1.)   
    random.seed(42)
    np.random.seed(42)
    key = jax.random.PRNGKey(42)
    rng = np.random.default_rng(np.asarray(key))
    np.random.RandomState(key)
    bp.math.random.seed(42)
    bm.random.seed(42)
    
    net = EINet()
    smt = SMT64bit(net)
    smt_result = smt.func_to_64bit_smt()

    npu_num = 1
    for npu_id in range(min(npu_num, 16)):
        hex_data = []
        for line in smt_result:
            instr_bin = "".join(line.bin_value_for_smt)
            parts = [instr_bin[i:i + 32] for i in range(0, len(instr_bin), 32)]
            hex_parts = [format(int(part, 2), '08X') for part in parts]
            hex_data.append(''.join(hex_parts))
        while len(hex_data) < 1024:
            hex_parts = [''.join(['0' * 16])]
            hex_data.extend(hex_parts)
            
    with open(f'smt_{npu_id}.hex', 'wt') as f_tmp:
        for item in hex_data:
            f_tmp.write(item + '\n')
            
    flag = 1