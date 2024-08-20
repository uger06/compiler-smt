from dataclasses import dataclass
from BrainpyAdapter.BrainpyBase import BrainpyBase
from .snn_compiler_96bit.common.smt_96_base import IBinary

@dataclass
class SMTBase:
    """ 硬件所需的一些SMT配置参数

    Args:
        tw (int, optional): _description_. Defaults to 5000000.
        step_max (int, optional): _description_. Defaults to 0.
        neu_num (int, optional): _description_. Defaults to 1023.
        rate (int, optional): _description_. Defaults to 0.
    """
    result_bits: int = 3                            
    """ SMT32bit的结果位宽"""
    
    tw: int = 5000000
    """ 时钟周期 """
    
    step_max: int = 0
    """ 仿真最大步数 """
    
    rate: int = 0
    """ 仿真速率 """
    
    neu_num_for32: int = 1023
    """ smt32bit的神经元数量"""
    
    rnd: int = 0
    """ 随机数种子 """
    
    cfg: int = 0                                    
    """ 配置参数 
    0: FP32, 2: INT16, 3: INT8
    """
    
    weight_phase: int = int(pow(2, 25))                     
    """ 权重累加时间窗 
    math.pow(2, 20), must be int 
    """
    
    ndma_phase:int = 32800
    """ NDMA 时间窗
    """
    
    total_step: int = 0
    """ 仿真总步长
    """
    
    _neu_num: int = 1024
    """ 神经元数量 """
    
    Ix : float = 2.0               
    """ 硬件固定输入电流
    """
    
    @property
    def neu_num(self) -> int:
    
        neu_bin = IBinary.dec2bin(self._neu_num, 11)
        cfg_bin = IBinary.dec2bin(self.cfg, 13 - 11)
        rnd_bin = IBinary.dec2bin(self.rnd, 17 - 13)
        rate_bin = IBinary.dec2bin(self.rate, 32 - 17)
        neu_num_bin = rate_bin + rnd_bin + cfg_bin + neu_bin
        
        self._neu_num = IBinary(0, 32)
        self._neu_num.bin_value = neu_num_bin
        self._neu_num = self._neu_num.dec_value

        return self._neu_num 
    
    @neu_num.setter
    def neu_num(self, value: int) -> int:
    
        neu_bin = IBinary.dec2bin(value, 11)
        cfg_bin = IBinary.dec2bin(self.cfg, 13 - 11)
        rnd_bin = IBinary.dec2bin(self.rnd, 17 - 13)
        rate_bin = IBinary.dec2bin(self.rate, 32 - 17)
        neu_num_bin = rate_bin + rnd_bin + cfg_bin + neu_bin
        
        self._neu_num = IBinary(0, 32)
        self._neu_num.bin_value = neu_num_bin
        self._neu_num = self._neu_num.dec_value

        return self._neu_num

if __name__ == '__main__':
    # smt = SMTBase()
    flag = 1
    
    SMTBase.neu_num()
    
    
    
    
    
    
    