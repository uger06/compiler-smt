"""SMT 96位指令
# @Author: uger
# @Date: 2024-07-08
"""

from __future__ import annotations

import re
from enum import auto
from typing import Union
from addict import Dict as AttrDict
from strenum import UppercaseStrEnum

from ..common.smt_64_reg import Register64, RegisterCollection
from ..common.asm_IEEE754 import IEEE754, IBinary
from ..common.smt_64_base import SMT64_FIELD_FORMAT


class OPType(UppercaseStrEnum):
    """4 位指令类型

    - 0: NOP
    - 1: SRAM_LOAD
    - 2: SRAM_SAVE
    - 3: BUS_LOAD
    - 4: BUS_SAVE
    - 5: ASSIGN_REG
    - 6: ASSIGN_IMM
    - 7: CALCU_REG
    - 8: CALCU_IMM
    - 9: VSET
    - 10: SPIKE_GEN
    """

    NOP = auto()
    """空操作: 0
    """

    SRAM_LOAD = auto()
    """SRAM 数据加载: 1
    """

    SRAM_SAVE = auto()
    """SRAM 数据保存: 2
    """
    
    BUS_LOAD = auto()
    """BUS 总线数据加载: 3
    """
    
    BUS_SAVE = auto()
    """BUS 总线数据保存: 4
    """

    ASSIGN_REG = auto()
    """寄存器赋值: 5
    """
    
    ASSIGN_IMM = auto()
    """立即数赋值: 6
    """
    
    CALCU_REG = auto()
    """寄存器计算: 7
    """
    
    CALCU_IMM = auto()
    """立即数计算: 8
    """
    
    VSET = auto()
    """膜电位重置: 9
    """

    SPIKE_GEN = auto()
    """脉冲发放: 10
    """

    @property
    def dec_value(self) -> int:
        """得到十进制数值

        Returns:
            int: 十进制数值
        """
        return {
            "NOP": 0,
            "SRAM_LOAD": 1,
            "SRAM_SAVE": 2,
            "BUS_LOAD": 3,
            "BUS_SAVE": 4,
            "ASSIGN_REG": 5,
            "ASSIGN_IMM": 6,
            "CALCU_REG": 7,
            "CALCU_IMM": 8,
            "VSET": 9,
            "SPIKE_GEN": 10,
        }[self.name]

    @property
    def operand(self) -> str:
        """二进制操作数

        >>> OPType.SRAM_LOAD.operand
        '00001'

        Returns:
            str: 二进制操作数
        """
        return IBinary.dec2bin(self.dec_value, 4)

class ALU_OPType(UppercaseStrEnum):
    """3 位指令代码
    
    - 0: ENABLE_OP
    - 1: NOP_OP
    - 2: ADD_OP
    - 3: MUL_OP
    - 4: EXP_OP
    - 5: IN_OP
    - 6: DIV_OP
    """

    ENABLE_OP = auto()
    """ 使能位: 0
    """

    NOP_OP = auto()
    """ 空操作: 1
    """
    
    ADD_OP = auto()
    """ 加法操作: 2
    """

    MUL_OP = auto()
    """ 乘法操作: 3
    """

    EXP_OP = auto()
    """ 指数操作: 4 
    """

    LN_OP = auto()
    """ 对数操作: 5
    """
    
    DIV_OP = auto()
    """ 除法操作: 6
    """
    
    @property
    def dec_value(self) -> int:
        """得到数值

        Returns:
            int: 数值
        """
        return {
            "ENABLE_OP": 0,
            "NOP_OP": 1,
            "ADD_OP": 2,
            "MUL_OP": 3,
            "EXP_OP": 4,
            "IN_OP": 5,
            "DIV_OP": 6,
        }[self.name]

    @property
    def operand(self) -> str:
        """二进制操作数

        >>> ALU_OpTtpe.ADD_OP.operand
        '10'

        Returns:
            str: 二进制操作数
        """
        return IBinary.dec2bin(self.dec_value, 3)


class ALUOUT_OPType(UppercaseStrEnum):
    """4 位指令代码

    - 0: add_S0
    - 1: add_S1
    - 2: mul_P0
    - 3: mul_P1
    - 4: exp_Z0
    - 5: exp_Z1
    - 6: in_Z0
    - 7: in_Z1
    - 8: div_Z0
    - 9: div_Z1
    - 10:alu_none
    """
    
    add_S0 = auto()
    """ 加法结果-S0: 0
    """
    

    add_S1 = auto()
    """ 加法结果-S1: 1
    """
    
    mul_P0 = auto()
    """ 乘法结果-P0: 2
    """

    mul_P1 = auto()
    """ 乘法结果-P1: 3
    """

    exp_Z0 = auto()
    """ 指数结果-Z0: 4
    """

    exp_Z1 = auto()
    """ 指数结果-Z1: 5
    """

    in_Z0 = auto()
    """ 对数结果-Z0: 6
    """

    in_Z1 = auto()
    """ 对数结果-Z1: 7
    """
    
    div_Z0 = auto()
    """ 除法结果-Z0: 8
    """

    div_Z1 = auto()
    """ 除法结果-Z1: 9
    """
    
    alu_none = auto()
    """ 空输出: 10
    """
    
    @property
    def dec_value(self) -> int:
        """得到数值

        Returns:
            int: 数值
        """
        return {
            "add_S0": 0,
            "add_S1": 1,
            "mul_P0": 2,
            "mul_P1": 3,
            "exp_Z0": 4,
            "exp_Z1": 5,
            "in_Z0": 6,
            "in_Z1": 7,
            "div_Z0": 8,
            "div_Z1": 9,
            "alu_none": 10,
        }[self.name]

    @property
    def operand(self) -> str:
        """二进制操作数

        >>> ALUOUT_OPType.add_S0.operand
        '0000'

        Returns:
            str: 二进制操作数
        """
        return IBinary.dec2bin(self.dec_value, 4)


class RS_OPType(UppercaseStrEnum):
    # FIXME, 确认占用位数
    """2 位指令代码
    
    - 0: NCU_ER_P
    - 1: NCU_ER_N
    - 2: NCU_SR_P
    - 3: ALU_OUT
    """

    NCU_ER_P = auto()
    """ 独享寄存器-正值: 0
    """

    NCU_ER_N = auto()
    """ 独享寄存器-负值: 1
    """
    
    NCU_SR_P = auto()
    """ 共享寄存器-正值: 2
    """

    ALU_OUT = auto()
    """ ALU 空输出: 3
    """

    @property
    def dec_value(self) -> int:
        """得到数值

        Returns:
            int: 数值
        """
        return {
            "NCU_ER_P": 0,
            "NCU_ER_N": 1,
            "NCU_SR_P": 2,
            "ALU_OUT": 3,
        }[self.name]

    @property
    def operand(self) -> str:
        """二进制操作数

        >>> RS_OPType.NCU_SR_P.operand
        '10'

        Returns:
            str: 二进制操作数
        """
        return IBinary.dec2bin(self.dec_value, 2)

class RD_OPType(UppercaseStrEnum):
    """1 位指令代码
    
    - 0: NCU_ER_RD_P
    - 1: NCU_SR_RD_P
    """

    NCU_ER_RD_P = auto()
    """ 独享结果寄存器-正值: 0
    """
    
    NCU_SR_RD_P = auto()
    """ 共享结果寄存器-正值: 1
    """

    @property
    def dec_value(self) -> int:
        """得到数值

        Returns:
            int: 数值
        """
        return {
            "NCU_ER_RD_P": 0,
            "NCU_SR_RD_P": 1,
        }[self.name]

    @property
    def operand(self) -> str:
        """二进制操作数

        >>> RD_OPType.NCU_SR_RD_P.operand
        '01'

        Returns:
            str: 二进制操作数
        """
        return IBinary.dec2bin(self.dec_value, 1)

class SMT_ASSIGN_I(IBinary):
    """SMT 赋值指令"""
    
    op_type: OPType
    """指令代码"""

    def __init__(self, 
                 OP_TYPE: OPType = OPType.ASSIGN_IMM,
                 NCU: int = 0,
                 IMM: Union[IEEE754, float] = 0.,
                 ALU_OP_0: ALU_OPType = ALU_OPType.ENABLE_OP,
                 RD_OP_0 : RD_OPType  = RD_OPType.NCU_SR_RD_P,
                 RD_REG_0: Register64 = RegisterCollection().SR_regs[0],
                 ALU_OP_1: ALU_OPType = ALU_OPType.NOP_OP,
                 RD_OP_1 : RD_OPType  = RD_OPType.NCU_SR_RD_P,
                 RD_REG_1: Register64 = RegisterCollection().SR_regs[0],
        ) -> None:
        """构造函数
        
        Args:
            OP_TYPE (OPType, optional): 指令代码. Defaults to OPType.ASSIGN_IMM.
            NCU (int, optional): NCU编号. Defaults to 0.
            IMM (Union[IEEE754, int], optional): 立即数. Defaults to 0.
            ALU_OP_0 (ALU_OPType, optional): ALU 运算类型. Defaults to ALU_OPType.ENABLE_OP.
            RD_OP_0 (RD_OPType, optional): 结果寄存器类型. Defaults to RD_OPType.NCU_SR_RD_P.
            RD_REG_0 (Register64, optional): 结果寄存器. Defaults to None.
            ALU_OP_1 (ALU_OPType, optional): ALU 运算类型. Defaults to ALU_OPType.ENABLE_OP.
            RD_OP_1 (RD_OPType, optional): 结果寄存器类型. Defaults to RD_OPType.NCU_SR_RD_P.
            RD_REG_1 (Register64, optional): 结果寄存器. Defaults to None.
        """
        super().__init__(dec_value=0, bin_width=64)
        self.op_type = OP_TYPE
        self.ncu = NCU
        self.imm = IMM
        self.alu_op_0 = ALU_OP_0
        self.rd_op_0 = RD_OP_0
        self.rd_reg_0 = RD_REG_0
        self.alu_op_1 = ALU_OP_1
        self.rd_op_1 = RD_OP_1  
        self.rd_reg_1 = RD_REG_1

    @property
    def arg_fields(self,) -> dict[str, Union[OPType, int, IEEE754, float, ALU_OPType, RD_OPType, Register64]]:
        """返回字段字典

        字段总宽度为64

        Returns:
            dict[str, Union[OPType, int, IEEE754, float, ALU_OPType, RD_OPType, Register64]]: _description_
        """
               
        return {
            "OP_TYPE": self.op_type,
            "NCU": self.ncu,
            "IMM": self.imm,
            "ALU_OP_0": self.alu_op_0,
            "RD_OP_0": self.rd_op_0,
            "RD_REG_0": self.rd_reg_0,
            "ALU_OP_1": self.alu_op_1,
            "RD_OP_1": self.rd_op_1,
            "RD_REG_1": self.rd_reg_1,
        }
        
        # return [self.op_type, self.ncu, self.imm, self.alu_op_0, self.rd_op_0, self.rd_reg_0, self.alu_op_1, self.rd_op_1, self.rd_reg_1]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}:{self.op_type} {self.imm}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self.dec_fields}>"

    @property
    def all_bin_fields(self) -> dict[str, str]:
        """返回一个包括所有SMT_ASSIGN_I字段在内的二进制字段字典

        Example:
            >>> f = SMT_ASSIGN_I(op_type=OPType.ASSIGN_IMM, )
            >>> f.all_bin_fields
            {'OP_TYPE': '0110', 
            'NCU': '00000001', 
            'IMM': '00111111100000000000000000000000', 
            'ALU_OP_0': '000', 'RD_OP_0': '1', 'RD_REG_0': '0000', 
            'ALU_OP_1': '001', 'RD_OP_1': '1', 'RD_REG_1': '0000'}

        Returns:
            dict[str, str]: 包括所有字段在内的二进制字段列表
        """
        
        # pattern = r"RD\\d+_REG"  # \\d+ 匹配一个或多个数字
        # re.search(r"RD\\d+_REG", name)
        
        result = dict()
        for name, field in self.arg_fields.items():
            if isinstance(field, Union[OPType, ALU_OPType, RD_OPType]):
                bin_value = field.operand
            elif isinstance(field, int) and name.startswith("NCU"):  # NCU编号 整数
                bin_value = IBinary.dec2bin(field, 8)
            elif isinstance(field, int) and name.startswith("RD_REG"):  # RD_REG编号 整数
                bin_value = IBinary.dec2bin(field, 4)
            elif isinstance(field, Union[float, int]) and name.startswith("IMM"):  # 立即数 整数
                bin_value = IEEE754(float(field)).bin_value
            elif isinstance(field, IEEE754):  # IEEE754 浮点数,
                bin_value = field.bin_value
            elif isinstance(field, Register64):  # 寄存器对象 length = 4
                bin_value = IBinary.dec2bin(field.index, 4)
            else:
                raise TypeError(f"{field = } ({type(field)}) 类型不对, 只能是 OPType, ALU_OPType, RD_OPType, int, float, IEEE754 或者 Register64")
            
            result[name] = bin_value
        return result

    @property
    def fields(self) -> dict[str, Union[OPType, int, IEEE754, float, ALU_OPType, RD_OPType, Register64]]:
        """字段字典

        - OPType: 指令代码
        - int: NCU编号 --> IBinary.dec2bin(NCU, 8)
        - IEEE754: 立即数
        - ALU_OPType: ALU 运算类型
        - RD_OPType: 结果寄存器类型
        - Register64: 结果寄存器

        Returns:
            Dict[str, Union[OPType, int, IEEE754, 
                            float, ALU_OPType, RD_OPType, Register64]]: 字段字典
        
        """
        result = dict()
        for name, field in self.arg_fields.items():
            if isinstance(field, Union[OPType, ALU_OPType, RD_OPType]):
                value = field.dec_value
            elif isinstance(field, int) and name.startswith("NCU"):  # NCU编号 整数
                value = field
            elif isinstance(field, Union[float, int]) and name.startswith("IMM"):  # 立即数 整数
                value = IEEE754(float(field))
            elif isinstance(field, IEEE754):  # IEEE754 浮点数,
                value = field
            elif isinstance(field, int) and name.startswith("RD_REG"):  # RD_REG编号 整数
                value = Register64(index=field)
            elif isinstance(field, Register64):  # 寄存器对象 length 必须是 4
                value = field  #IBinary.dec2bin(field.index, 4)
            else:
                raise TypeError(f"{field = } ({type(field)}) 类型不对, 只能是 OPType, ALU_OPType, RD_OPType, int, float, IEEE754 或者 Register64")
            
            result[name] = value
        return result

    @property
    def bin_fields(self) -> str:
        """返回指令确定位置的64位二进制比特流
        
        [ opcode, reserved, ncu_assign, imm, rd[0], rd[1] ] 
        ---10--------4----------8--------32----5------5---
        
        Returns:
            list[str]: 指令确定位置的64位二进制比特流
        
        """
        
        bin_fields =  AttrDict()
        bin_fields.opcode = IBinary((self.fields['OP_TYPE'] << 6) + \
                (self.fields['ALU_OP_0'] << 3) + \
                (self.fields['ALU_OP_1'] << 0 ), 10).bin_value
        bin_fields.reserved = IBinary(0, 4).bin_value
        bin_fields.ncu_assign = IBinary(self.fields['NCU'], 8).bin_value
        bin_fields.imm = self.fields['IMM'].bin_value
        bin_fields.rd_0 = IBinary((self.fields['RD_OP_0'] << 4) + \
                (self.fields['RD_REG_0'].index << 0), 5).bin_value
        bin_fields.rd_1 = IBinary((self.fields['RD_OP_1'] << 4) + \
                (self.fields['RD_REG_1'].index << 0), 5).bin_value
        
        self.bin_value = "".join(bin_fields.values())
        
        return bin_fields

    @property
    def dec_fields(self) -> list[int]:
        """返回一个不包括空字段在内的整数字段列表

        - 注意: 不能区别常数数值和寄存器编号
        - 请用 `fields` 属性区别常数数值和寄存器编号

        Example:
            >>> f = SMT_ASSIGN_I(op_type=OPType.ASSIGN_IMM, )
            >>> f.bin_value
            '0110000001000000000001001111111000000000000000000000001000010000'
            >>> f.bin_fields
            {'opcode': '0110000001', 'reserved': '0000', 'ncu_assign': '00000001', 
            'imm': '00111111100000000000000000000000', 
            'rd_0': '10000', 'rd_1': '10000'}
            >>> f.dec_fields
            [6, 1, 1065353216, 0, 1, 0, 1, 1, 0]

        Returns:
            list[IEEE754, int]: 包含十进制字段的列表
        """
        return [IBinary.bin2dec(bin_value) for bin_value in self.all_bin_fields.values()]

    @property
    def bin_value_for_smt(self) -> str:
        """返回 指令 64bit二进制

        Example:
            >>> f = SMT_ASSIGN_I(op_type=OPType.ASSIGN_IMM, )
            >>> f.bin_value
            '0110000001000000000001001111111000000000000000000000001000010000'
            >>> f.bin_value_for_smt
            '0110000001000000000001001111111000000000000000000000001000010000'

        Returns:
            str: 64bit二进制操作数
        """
        
        bin_fields =  AttrDict()
        bin_fields.opcode = IBinary((self.op_type.dec_value << 6) + \
                (self.alu_op_0.dec_value << 3) + \
                (self.alu_op_1.dec_value << 0 ), 10).bin_value
        bin_fields.reserved = IBinary(0, 4).bin_value
        bin_fields.ncu_assign = IBinary(self.ncu, 8).bin_value
        bin_fields.imm = IEEE754(float(self.imm)).bin_value
        bin_fields.rd_0 = IBinary((self.rd_op_0.dec_value << 4) + \
                (self.rd_reg_0.index << 0), 5).bin_value
        bin_fields.rd_1 = IBinary((self.rd_op_1.dec_value << 4) + \
                (self.rd_reg_1.index << 0), 5).bin_value
        
        result = "".join(bin_fields.values())
        
        return result


class SMT_RW(IBinary):
    """SMT 读写指令"""
    
    op_type: OPType
    """指令代码"""
    
    def __init__(self, 
                 OP_TYPE: OPType = OPType.NOP,
                 ) -> None:
        """构造函数
        
        Args:
            OP_TYPE (OPType, optional): 指令代码. Defaults to OPType.SRAM_LOAD.
        """
        super().__init__(dec_value=0, bin_width=64)
        self.op_type = OP_TYPE

    def __str__(self) -> str:
        return f"{self.__class__.__name__}:{self.op_type}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self.dec_fields}>"

    @property
    def arg_fields(self,) -> dict[str, OPType]:
        """返回字段字典

        字段总宽度为64

        Returns:
            dict[str, OPType]: 字段字典
        """
               
        return {
            "OP_TYPE": self.op_type,
        }

    @property
    def all_bin_fields(self) -> dict[str, str]:
        """返回一个包括所有SMT_ASSIGN_I字段在内的二进制字段字典

        Example:
            >>> f = SMT_ASSIGN_I(op_type=OPType.NOP, )
            >>> f.all_bin_fields
            {'OP_TYPE': '0110',}

        Returns:
            dict[str, str]: 包括所有字段在内的二进制字段列表
        """
        result = dict()
        for name, field in self.arg_fields.items():
            if isinstance(field, OPType):
                bin_value = field.operand
            else:
                raise TypeError(f"{field = } ({type(field)}) 类型不对, 只能是 OPType")
            
            result[name] = bin_value
        return result

    @property
    def fields(self) -> dict[str, OPType]:
        """字段字典

        - OPType: 指令代码

        Returns:
            Dict[str, OPType]: 字段字典
        """
        result = dict()
        for name, field in self.arg_fields.items():
            if isinstance(field, OPType):
                value = field.dec_value
            else:
                raise TypeError(f"{field = } ({type(field)}) 类型不对, 只能是 OPType")
            
            result[name] = value
        return result


    @property
    def bin_fields(self) -> str:
        """返回指令确定位置的64位二进制比特流
        
        [ opcode, reserved ] 
        ----4------60------
        
        Returns:
            list[str]: 指令确定位置的64位二进制比特流
        
        """
        
        bin_fields =  AttrDict()
        bin_fields.opcode = IBinary((self.fields['OP_TYPE']), 4).bin_value
        bin_fields.reserved = IBinary(0, 60).bin_value
        
        self.bin_value = "".join(bin_fields.values())
        
        return bin_fields
        
    @property
    def dec_fields(self) -> list[int]:
        """返回一个不包括空字段在内的整数字段列表

        - 注意: 不能区别常数数值和寄存器编号
        - 请用 `fields` 属性区别常数数值和寄存器编号

        Returns:
            list[int]: 包含十进制字段的列表
        """
        return [IBinary.bin2dec(bin_value) for bin_value in self.all_bin_fields.values()]

    @property
    def bin_value_for_smt(self) -> str:
        """返回 指令 64bit二进制

        Example:
            >>> f = SMT_RW(op_type=OPType.NOP, )
            >>> f.bin_value
            '0110000001000000000001001111111000000000000000000000001000010000'
            >>> f.bin_value_for_smt
            '0110000001000000000001001111111000000000000000000000001000010000'

        Returns:
            str: 64bit二进制操作数
        """
        
        bin_fields =  AttrDict()
        bin_fields.opcode = IBinary(self.op_type.dec_value, 4).bin_value
        bin_fields.reserved = IBinary(0, 60).bin_value
        
        result = "".join(bin_fields.values())
        
        return result


class SMT_RC(IBinary):
    """SMT 寄存器计算指令"""
    
    op_type: OPType
    """指令代码"""
    
    def __init__(self,
                OP_TYPE: OPType = OPType.CALCU_REG,
                
                ALU1_OP: ALU_OPType = ALU_OPType.ADD_OP,
                RS1_OP_0 : RS_OPType  = RS_OPType.NCU_ER_P,
                RS1_REG_0: Register64 = RegisterCollection().regs[0],
                RS1_OP_1 : RS_OPType  = RS_OPType.NCU_ER_P,
                RS1_REG_1: Register64 = RegisterCollection().regs[0],
                RD1_OP : RD_OPType  = RD_OPType.NCU_ER_RD_P,
                RD1_REG: Register64 = RegisterCollection().regs[0],
                
                ALU2_OP: ALU_OPType = ALU_OPType.NOP_OP,
                RS2_OP_0 : RS_OPType  = RS_OPType.ALU_OUT,
                RS2_REG_0: Register64 = RegisterCollection().regs[10],  # ALUOUT_OPType.alu_none,
                RS2_OP_1 : RS_OPType  = RS_OPType.ALU_OUT,
                RS2_REG_1: Register64 = RegisterCollection().regs[10],  # ALUOUT_OPType.alu_none,
                RD2_OP : RD_OPType  = RD_OPType.NCU_ER_RD_P,
                RD2_REG: Register64 = RegisterCollection().regs[0],
        ) -> None:
        """构造函数
        
        Args:
            OP_TYPE (OPType, optional): 指令代码. Defaults to OPType.CALCU_REG.
            ALU1_OP (ALU_OPType, optional): ALU 运算类型. Defaults to ALU_OPType.ADD_OP.
            RS1_OP_0 (RS_OPType, optional): RS1 操作类型. Defaults to RS_OPType.NCU_ER_P.
            RS1_REG_0 (Register64, optional): RS1 寄存器. Defaults to RegisterCollection().regs[0].
            RS1_OP_1 (RS_OPType, optional): RS1 操作类型. Defaults to RS_OPType.NCU_ER_P.
            RS1_REG_1 (Register64, optional): RS1 寄存器. Defaults to RegisterCollection().regs[0].
            RD1_OP (RD_OPType, optional): RD1 操作类型. Defaults to RD_OPType.NCU_ER_RD_P.
            RD1_REG (Register64, optional): RD1 寄存器. Defaults to RegisterCollection().regs[0].
            ALU2_OP (ALU_OPType, optional): ALU 运算类型. Defaults to ALU_OPType.NOP_OP.
            RS2_OP_0 (RS_OPType, optional): RS2 操作类型. Defaults to RS_OPType.ALU_OUT.
            RS2_REG_0 (Register64, optional): RS2 寄存器. Defaults to ALUOUT_OPType.alu_none,
            RS2_OP_1 (RS_OPType, optional): RS2 操作类型. Defaults to RS_OPType.ALU_OUT.
            RS2_REG_1 (Register64, optional): RS2 寄存器. Defaults to ALUOUT_OPType.alu_none,
            RD2_OP (RD_OPType, optional): RD2 操作类型. Defaults to RD_OPType.NCU_ER_RD_P.
            RD2_REG (Register64, optional): RD2 寄存器. Defaults to RegisterCollection().regs[0].
        """
        super().__init__(dec_value=0, bin_width=64)
        self.op_type = OP_TYPE
        self.alu1_op = ALU1_OP
        self.rs1_op_0 = RS1_OP_0
        self.rs1_reg_0 = RS1_REG_0
        self.rs1_op_1 = RS1_OP_1
        self.rs1_reg_1 = RS1_REG_1
        self.rd1_op = RD1_OP
        self.rd1_reg = RD1_REG
        
        self.alu2_op = ALU2_OP
        self.rs2_op_0 = RS2_OP_0
        self.rs2_reg_0 = RS2_REG_0
        self.rs2_op_1 = RS2_OP_1
        self.rs2_reg_1 = RS2_REG_1
        self.rd2_op = RD2_OP
        self.rd2_reg = RD2_REG

    def __str__(self) -> str:
        return f"{self.__class__.__name__}:{self.op_type}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self.dec_fields}>"
    
    @property
    def arg_fields(self,) -> dict[str, Union[OPType, ALU_OPType, RS_OPType, RD_OPType, Register64]]:
        """返回字段字典

        字段总宽度为64

        Returns:
            dict[str, Union[OPType, ALU_OPType, RS_OPType, RD_OPType, Register64]]: 字段字典
        """
               
        return {
            "OP_TYPE": self.op_type,
            "ALU1_OP": self.alu1_op,
            "RS1_OP_0": self.rs1_op_0,
            "RS1_REG_0": self.rs1_reg_0,    
            "RS1_OP_1": self.rs1_op_1,
            "RS1_REG_1": self.rs1_reg_1,
            "RD1_OP": self.rd1_op,
            "RD1_REG": self.rd1_reg,
            
            "ALU2_OP": self.alu2_op,
            "RS2_OP_0": self.rs2_op_0,
            "RS2_REG_0": self.rs2_reg_0,
            "RS2_OP_1": self.rs2_op_1,
            "RS2_REG_1": self.rs2_reg_1,
            "RD2_OP": self.rd2_op,
            "RD2_REG": self.rd2_reg,
        }
        

    @property
    def all_bin_fields(self) -> dict[str, str]:
        """返回一个包括所有SMT_RC字段在内的二进制字段字典

        Example:
            >>> f = SMT_RC(OP_TYPE=OPType.CALCU_REG, )
            >>> f.all_bin_fields
        
        Returns:
            dict[str, str]: 包括所有字段在内的二进制字段列表
        """
        result = dict()
        for name, field in self.arg_fields.items():
            if isinstance(field, Union[OPType, ALU_OPType, RS_OPType, RD_OPType]):
                bin_value = field.operand
            elif isinstance(field, Register64):  
                bin_value = IBinary.dec2bin(field.index, 4)
            elif isinstance(field, int) and re.search([r"RD\\d+_REG", r"RS\\d+_REG"], name):  # REG编号 整数
                bin_value = IBinary.dec2bin(field, 4)
            else:
                raise TypeError(f"{field = } ({type(field)}) 类型不对, 只能是 OPType, ALU_OPType, RS_OPType, RD_OPType, int或者 Register64")
            
            result[name] = bin_value
        return result

    @property
    def fields(self) -> dict[str, Union[OPType, ALU_OPType, RS_OPType, RD_OPType, Register64]]:
        """字段字典

        - OPType: 指令代码
        - ALU_OPType: ALU 运算类型
        - RS_OPType: RS 操作类型
        - RD_OPType: RD 操作类型
        - Register64: 寄存器

        Returns:
            Dict[str, Union[OPType, ALU_OPType, RS_OPType, RD_OPType, Register64]]: 字段字典
        """
        result = dict()
        for name, field in self.arg_fields.items():
            if isinstance(field, Union[OPType, ALU_OPType, RS_OPType, RD_OPType]):
                value = field.dec_value
            elif isinstance(field, Register64):
                value = field
            elif isinstance(field, int) and re.search([r"RD\\d+_REG", r"RS\\d+_REG"], name):  # REG编号 整数
                value = Register64(index=field)
            else:
                raise TypeError(f"{field = } ({type(field)}) 类型不对, 只能是 OPType, ALU_OPType, RS_OPType, RD_OPType, int或者 Register64")

            result[name] = value
        return result

    @property
    def bin_fields(self) -> str:
        """返回指令确定位置的64位二进制比特流
        
        [ opcode, reserved, [rs1_op_0, rs1_reg_0],  [rs2_op_0, rs2_reg_0], [rs1_op_1, rs1_reg_1], [rs2_op_1, rs2_reg_1], 
        ---10--------20--------------6------------------------6----------------------6--------------------6----------
        
       [rd1_op, rd1_reg], [rd2_op, rd2_reg] ] 
        --------5-----------------5--------
        
        Returns:
            list[str]: 指令确定位置的64位二进制比特流
        
        """
        
        bin_fields =  AttrDict()
        bin_fields.opcode =IBinary((self.fields['OP_TYPE'] << 6) + \
                (self.fields['ALU1_OP'] << 3) + \
                (self.fields['ALU2_OP'] << 0 ), 10).bin_value
        
        bin_fields.reserved = IBinary(0, 20).bin_value
        
        bin_fields.rs1_0 = IBinary((self.fields['RS1_OP_0'] << 4) + \
                (self.fields['RS1_REG_0'].index << 0), 6).bin_value
        
        bin_fields.rs2_0 = IBinary((self.fields['RS2_OP_0'] << 4) + \
                (self.fields['RS2_REG_0'].index << 0), 6).bin_value
        
        bin_fields.rs1_1 = IBinary((self.fields['RS1_OP_1'] << 4) + \
                (self.fields['RS1_REG_1'].index << 0), 6).bin_value
        
        bin_fields.rs2_1 = IBinary((self.fields['RS2_OP_1'] << 4) + \
                (self.fields['RS2_REG_1'].index << 0), 6).bin_value
        
        bin_fields.rd1 = IBinary((self.fields['RD1_OP'] << 4) + \
                (self.fields['RD1_REG'].index << 0), 5).bin_value
        
        bin_fields.rd2 = IBinary((self.fields['RD2_OP'] << 4) + \
                (self.fields['RD2_REG'].index << 0), 5).bin_value
        
        self.bin_value = "".join(bin_fields.values())
        
        return bin_fields
        
    @property
    def dec_fields(self) -> list[int]:
        """返回一个不包括空字段在内的整数字段列表

        - 注意: 不能区别常数数值和寄存器编号
        - 请用 `fields` 属性区别常数数值和寄存器编号

        Returns:
            list[int]: 包含十进制字段的列表
        """
        return [IBinary.bin2dec(bin_value) for bin_value in self.all_bin_fields.values()]

    @property
    def bin_value_for_smt(self) -> str:
        """返回 指令 64bit二进制

        Example:
            >>> f = SMT_RC(op_type=OPType.CALCU_REG, )
            >>> f.bin_value
            '0110000001000000000001001111111000000000000000000000001000010000'
            >>> f.bin_value_for_smt
            '0110000001000000000001001111111000000000000000000000001000010000'

        Returns:
            str: 64bit二进制操作数
        """
        
        bin_fields =  AttrDict()
        bin_fields.opcode = IBinary((self.op_type.dec_value << 6) + \
                (self.alu1_op.dec_value << 3) + \
                (self.alu2_op.dec_value << 0 ), 10).bin_value
        
        bin_fields.reserved = IBinary(0, 20).bin_value
        
        bin_fields.rs1_0 = IBinary((self.rs1_op_0.dec_value << 4) + \
                (self.rs1_reg_0.index << 0), 6).bin_value
        
        bin_fields.rs2_0 = IBinary((self.rs2_op_0.dec_value << 4) + \
                (self.rs2_reg_0.index << 0), 6).bin_value
        
        bin_fields.rs1_1 = IBinary((self.rs1_op_1.dec_value << 4) + \
                (self.rs1_reg_1.index << 0), 6).bin_value
        
        bin_fields.rs2_1 = IBinary((self.rs2_op_1.dec_value << 4) + \
                (self.rs2_reg_1.index << 0), 6).bin_value
        
        bin_fields.rd1 = IBinary((self.rd1_op.dec_value << 4) + \
                (self.rd1_reg.index << 0), 5).bin_value
        
        bin_fields.rd2 = IBinary((self.rd2_op.dec_value << 4) + \
                (self.rd2_reg.index << 0), 5).bin_value
        
        result = "".join(bin_fields.values())
        
        return result


class SMT_IC(IBinary):
    """SMT 常数计算指令"""
    
    ope_type: OPType
    """指令代码"""
    
    def __init__(self, 
                 OP_TYPE: OPType = OPType.CALCU_IMM,
                 IMM: Union[IEEE754, float] = 0.,
                 
                 ALU1_OP: ALU_OPType = ALU_OPType.ADD_OP,
                 RS1_OP: RS_OPType = RS_OPType.NCU_ER_P,
                 RS1_REG: Register64 = RegisterCollection().regs[0],
                 RD1_OP: RD_OPType = RD_OPType.NCU_ER_RD_P,
                 RD1_REG: Register64 = RegisterCollection().regs[0],
                 
                 ALU2_OP: ALU_OPType = ALU_OPType.ADD_OP,
                 RS2_OP: RS_OPType = RS_OPType.NCU_ER_P,
                 RS2_REG: Register64 = RegisterCollection().regs[0],
                 RD2_OP: RD_OPType = RD_OPType.NCU_ER_RD_P,
                 RD2_REG: Register64 = RegisterCollection().regs[0],
        ) -> None:
        """构造函数
        
        Args:
            OP_TYPE (OPType, optional): 指令代码. Defaults to OPType.CALCU_IMM.
            IMM (Union[IEEE754, float], optional): 常数. Defaults to 0..
            ALU1_OP (ALU_OPType, optional): ALU 运算类型. Defaults to ALU_OPType.ADD_OP.
            RS1_OP (RS_OPType, optional): RS1 操作类型. Defaults to RS_OPType.NCU_ER_P.
            RS1_REG (Register64, optional): RS1 寄存器. Defaults to RegisterCollection().regs[0].
            RD1_OP (RD_OPType, optional): RD1 操作类型. Defaults to RD_OPType.NCU_ER_RD_P.
            RD1_REG (Register64, optional): RD1 寄存器. Defaults to RegisterCollection().regs[0].
            ALU2_OP (ALU_OPType, optional): ALU 运算类型. Defaults to ALU_OPType.ADD_OP.
            RS2_OP (RS_OPType, optional): RS2 操作类型. Defaults to RS_OPType.NCU_ER_P.
            RS2_REG (Register64, optional): RS2 寄存器. Defaults to RegisterCollection().regs[0].
            RD2_OP (RD_OPType, optional): RD2 操作类型. Defaults to RD_OPType.NCU_ER_RD_P.
            RD2_REG (Register64, optional): RD2 寄存器. Defaults to RegisterCollection().regs[0].
        """
        super().__init__(dec_value=0, bin_width=64)
        self.op_type = OP_TYPE
        self.imm = IMM
        self.alu1_op = ALU1_OP
        self.rs1_op = RS1_OP
        self.rs1_reg = RS1_REG
        self.rd1_op = RD1_OP
        self.rd1_reg = RD1_REG
        
        self.alu2_op = ALU2_OP
        self.rs2_op = RS2_OP
        self.rs2_reg = RS2_REG
        self.rd2_op = RD2_OP
        self.rd2_reg = RD2_REG

    def __str__(self) -> str:
        return f"{self.__class__.__name__}:{self.op_type}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self.dec_fields}>"
    
    
    @property
    def arg_fields(self,) -> dict[str, Union[OPType, ALU_OPType, RS_OPType, RD_OPType, Register64, int, float, IEEE754]]:
        """返回字段字典

        字段总宽度为64

        Returns:
            dict[str, Union[OPType, ALU_OPType, RS_OPType, RD_OPType, Register64, int, float, IEEE754]]: 字段字典
        """
               
        return {
            "OP_TYPE": self.op_type,
            "IMM": self.imm,
            "ALU1_OP": self.alu1_op,
            "RS1_OP": self.rs1_op,
            "RS1_REG": self.rs1_reg,    
            "RD1_OP": self.rd1_op,
            "RD1_REG": self.rd1_reg,
            
            "ALU2_OP": self.alu2_op,
            "RS2_OP": self.rs2_op,
            "RS2_REG": self.rs2_reg,
            "RD2_OP": self.rd2_op,
            "RD2_REG": self.rd2_reg,
        }
        

    @property
    def all_bin_fields(self) -> dict[str, str]:
        """返回一个包括所有SMT_ASSIGN_I字段在内的二进制字段字典

        Example:
            >>> f = SMT_IC(OP_TYPE=OPType.CALCU_IMM, IMM=1.23456789)
            >>> f.all_bin_fields
        
        Returns:
            dict[str, str]: 包括所有字段在内的二进制字段列表
        """
        result = dict()
        for name, field in self.arg_fields.items():
            if isinstance(field, Union[OPType, ALU_OPType, RS_OPType, RD_OPType]):
                bin_value = field.operand
            elif isinstance(field, Union[float, int]) and name.startswith("IMM"):  # 立即数 整数
                bin_value = IEEE754(float(field)).bin_value
            elif isinstance(field, IEEE754):
                bin_value = field.bin_value
            elif isinstance(field, Register64):  
                bin_value = IBinary.dec2bin(field.index, 4)
            elif isinstance(field, int) and re.search([r"RD\\d+_REG", r"RS\\d+_REG"], name):  # REG编号 整数
                bin_value = IBinary.dec2bin(field, 4)
            else:
                raise TypeError(f"{field = } ({type(field)}) 类型不对, 只能是 OPType, ALU_OPType, RS_OPType, RD_OPType, float, IEEE754, int或者 Register64")

            result[name] = bin_value
        return result

    @property
    def fields(self) -> dict[str, Union[OPType, ALU_OPType, RS_OPType, RD_OPType, Register64, float, IEEE754]]:
        """字段字典

        - OPType: 指令代码
        - ALU_OPType: ALU 运算类型
        - RS_OPType: RS 操作类型
        - RD_OPType: RD 操作类型
        - Register64: 寄存器
        - float: 常数浮点数
        - IEEE754: 常数

        Returns:
            Dict[str, Union[OPType, ALU_OPType, RS_OPType, RD_OPType, Register64, float, IEEE754]]: 字段字典
        """
        result = dict()
        for name, field in self.arg_fields.items():
            if isinstance(field, Union[OPType, ALU_OPType, RS_OPType, RD_OPType]):
                value = field.dec_value
            elif isinstance(field, Union[float, int]) and name.startswith("IMM"):  # 立即数 整数
                value = IEEE754(float(field))
            elif isinstance(field, IEEE754):
                value = field
            elif isinstance(field, Register64):
                value = field
            elif isinstance(field, int) and re.search([r"RD\\d+_REG", r"RS\\d+_REG"], name):  # REG编号 整数
                value = Register64(index=field)
            else:
                raise TypeError(f"{field = } ({type(field)}) 类型不对, 只能是 OPType, ALU_OPType, RS_OPType, RD_OPType, float, IEEE754, int或者 Register64")

            result[name] = value
        return result

    @property
    def bin_fields(self) -> str:
        """返回指令确定位置的64位二进制比特流
        
        [ opcode, imm, [rs1_op, rs1_reg], [rs2_op, rs2_reg], [rd1_op, rd1_reg], [rd2_op, rd2_reg] ] 
        ---10------32----------6-------------------6-----------------5------------------5---------
        
        Returns:
            list[str]: 指令确定位置的64位二进制比特流
        
        """
        
        bin_fields =  AttrDict()
        bin_fields.opcode =IBinary((self.fields['OP_TYPE'] << 6) + \
                (self.fields['ALU1_OP'] << 3) + \
                (self.fields['ALU2_OP'] << 0 ), 10).bin_value

        bin_fields.imm = self.fields['IMM'].bin_value
        
        bin_fields.rs1_0 = IBinary((self.fields['RS1_OP'] << 4) + \
                (self.fields['RS1_REG'].index << 0), 6).bin_value
        
        bin_fields.rs2_0 = IBinary((self.fields['RS2_OP'] << 4) + \
                (self.fields['RS2_REG'].index << 0), 6).bin_value
        
        bin_fields.rd1 = IBinary((self.fields['RD1_OP'] << 4) + \
                (self.fields['RD1_REG'].index << 0), 5).bin_value
        
        bin_fields.rd2 = IBinary((self.fields['RD2_OP'] << 4) + \
                (self.fields['RD2_REG'].index << 0), 5).bin_value
        
        self.bin_value = "".join(bin_fields.values())
        
        return bin_fields
        
    @property
    def dec_fields(self) -> list[int]:
        """返回一个不包括空字段在内的整数字段列表

        - 注意: 不能区别常数数值和寄存器编号
        - 请用 `fields` 属性区别常数数值和寄存器编号

        Returns:
            list[int]: 包含十进制字段的列表
        """
        return [IBinary.bin2dec(bin_value) for bin_value in self.all_bin_fields.values()]
    
    @property
    def bin_value_for_smt(self) -> str:
        """返回 指令 64bit二进制

        Example:
            >>> f = SMT_IC(op_type=OPType.CALCU_IMM, )
            >>> f.bin_value
            '0110000001000000000001001111111000000000000000000000001000010000'
            >>> f.bin_value_for_smt
            '0110000001000000000001001111111000000000000000000000001000010000'

        Returns:
            str: 64bit二进制操作数
        """
        
        bin_fields =  AttrDict()
        bin_fields.opcode =IBinary((self.op_type.dec_value << 6) + \
                (self.alu1_op.dec_value << 3) + \
                (self.alu2_op.dec_value << 0 ), 10).bin_value

        bin_fields.imm = IEEE754(float(self.imm)).bin_value
        
        bin_fields.rs1_0 = IBinary((self.rs1_op.dec_value << 4) + \
                (self.rs1_reg.index << 0), 6).bin_value
        
        bin_fields.rs2_0 = IBinary((self.rs2_op.dec_value << 4) + \
                (self.rs2_reg.index << 0), 6).bin_value
        
        bin_fields.rd1 = IBinary((self.rd1_op.dec_value << 4) + \
                (self.rd1_reg.index << 0), 5).bin_value
        
        bin_fields.rd2 = IBinary((self.rd2_op.dec_value << 4) + \
                (self.rd2_reg.index << 0), 5).bin_value
        
        result = "".join(bin_fields.values())
        
        return result
    
    
class CalField(IBinary):
    """支持CALCU_IMM, CALCU_REG的54位操作字段
   
    - `op_code` (OPCode): 指令代码
    - `field_format` (list[int]): 字段格式
    - `bin_value` (str):  54 位二进制操作数
    - `operand ` (str):  54 位二进制操作数-别称
    - `bin_fields` (list[str]): 字段列表, 二进制格式
    - `fields` (list[IEEE754, int]): 字段列表:
        - `int`: 寄存器编号
        - `IEEE754`: 常数

    Example:
        >>> f = CalField(op_code=OPType.CALCU_REG, fields=[0,10, 0,10, 0,6, 0,10, 0,10, 0,9])
        >>> f
        [0, 10, 0, 10, 0, 6, 0, 10, 0, 10, 0, 9]
        >>> f.field_format
        [-20, 2, 4, 2, 4, 1, 4, 2, 4, 2, 4, 1, 4]
        >>> f.bin_fields
        ['00', '1010', '00', '1010', '0', '0110', '00', '1010', '00', '1010', '0', '1001']
        >>> f.fields
        [0, 10, 0, 10, 0, 6, 0, 10, 0, 10, 0, 9]
        >>> f.bin_value
        '000000000000000000000010101010101111100001'
    """

    op_code: OPType
    """指令代码
    - 7: CALCU_REG
    - 8: CALCU_IMM
    """

    raw_fields: list[Union[IEEE754, int, str, float]]
    """原始字段列表"""

    @property
    def field_format(self) -> list[int]:
        """字段格式, e.g. `"CALCU_IMM": [32, 2,4, 1,4, 2,4, 1,4]`

        Returns:
            list[int]: 字段格式
        """
        if self.op_code not in SMT64_FIELD_FORMAT:
            raise NotImplementedError(f"不支持的 {self.op_code.name = }")
        return SMT64_FIELD_FORMAT[self.op_code.name]

    def __init__(
        self,
        op_code: OPType = OPType.CALCU_REG,
        fields: list[IEEE754, int, float] = None,
    ) -> None:
        """构造函数

        Args:
            op_code (OPCode): 操作类型代码, 默认为 `OPType.CALCU_REG`
            fields (list[Union[IEEE754, int, float]]): 子字段列表
                int: 寄存器编号,
                IEEE754: 常数浮点数,
                float: 常数浮点数
        """
        super().__init__(dec_value=0, bin_width=54)
        self.op_code = op_code
        if fields is None:
            self.raw_fields = self.bin_fields
        self.fields = fields

    def __str__(self) -> str:
        return self.fields

    def __repr__(self) -> str:
        return f'{self.op_code.name}: {str(self.fields)}'

    @property
    def all_bin_fields(self) -> list[str]:
        """返回一个包括空字段在内的二进制字段列表

        Example:
            >>> f = CalField(op_code=OPType.CALCU_REG, fields=[0,10, 0,10, 0,6, 0,10, 0,10, 0,9])
            >>> f.all_bin_fields
            ['00000000000000000000', '00', '1010', '00', '1010', '0', '0110', '00', '1010', '00', '1010', '0', '1001']
            >>> f.bin_fields         
            ['00', '1010', '00', '1010', '0', '0110', '00', '1010', '00', '1010', '0', '1001']
            >>> f = CalField(op_code=OPType.CALCU_IMM, fields=[35.6, 0,10, 0,6, 0,9, 0,7])
            >>> f.all_bin_fields
            ['01000010000011100110011001100110', '00', '1010', '0', '0110', '00', '1001', '0', '0111']
            >>> f.bin_fields         
            ['01000010000011100110011001100110', '00', '1010', '0', '0110', '00', '1001', '0', '0111']

        Returns:
            list[str]: 包括空字段在内的二进制字段列表
        """
        result: list[str] = []
        start = 0
        bin_value = self.bin_value
        for width in self.field_format:
            width = abs(width)
            result.append(bin_value[start : start + width])
            start += width
        return result

    @property
    def bin_fields(self) -> list[str]:
        """返回一个不包括空字段在内的二进制字段列表

        Example:
            >>> f = CalField(op_code=OPType.CALCU_REG, fields=[0,10, 0,10, 0,6, 0,10, 0,10, 0,9])
            >>> f.bin_fields         
            ['00', '1010', '00', '1010', '0', '0110', '00', '1010', '00', '1010', '0', '1001']
            
            >>> f = CalField(op_code=OPType.CALCU_IMM, fields=[35.6, 0,10, 0,6, 0,9, 0,7])
            >>> f.bin_fields         
            ['01000010000011100110011001100110', '00', '1010', '0', '0110', '00', '1001', '0', '0111']

        Returns:
            list[str]: 不包括空字段在内的二进制字段列表
        """
        result: list[str] = []
        for length, bin_value in zip(self.field_format, self.all_bin_fields):
            if 0 < length <= 32:  # 跳过空字段
                result.append(bin_value)
        return result

    @property
    def dec_fields(self) -> list[int]:
        """返回一个不包括空字段在内的整数字段列表

        - 注意: 不能区别常数数值和寄存器编号
        - 请用 `fields` 属性区别常数数值和寄存器编号

        Example:
            >>> f = CalField(op_code=OPType.CALCU_IMM, fields=[21.3, 0,10, 0,6, 0,9, 0,7])
            >>> f.field_format
            [32, 2, 4, 1, 4, 2, 4, 1, 4]
            >>> f.dec_fields
            [1101686374, 0, 10, 0, 6, 0, 9, 0, 7]

        Returns:
            list[IEEE754, int]: 包含十进制字段的列表
        """
        return [IBinary.bin2dec(bin_value) for bin_value in self.bin_fields]

    @property
    def fields(self) -> list[int]:
        """字段列表, 不包括空字段

        - int: 寄存器编号
        - IEEE754: 常数浮点数

        Example:
            >>> f = CalField(op_code=OPType.CALCU_IMM, fields=[21.3, 0,10, 0,6, 0,9, 0,7])
            >>> f.field_format
            [32, 2, 4, 1, 4, 2, 4, 1, 4]
            >>> f.fields
            [<IEEE754:1101686374:21.30>, 0, 10, 0, 6, 0, 9, 0, 7]
            >>> f.fields = [35.6, 0,10, 0,6, 0,9, 0,7]
            >>> f.fields
            [<IEEE754:1108239974:35.60>, 0, 10, 0, 6, 0, 9, 0, 7]

        Returns:
            list[IEEE754, int]: 字段列表
        """
        result: list[Union[IEEE754, int]] = []
        for bin_value in self.bin_fields:
            if len(bin_value) == 32:  # REVIEW: 32 位一定是 IEEE754 浮点数
                result.append(IEEE754(bin_value))
                continue
            result.append(IBinary.bin2dec(bin_value))
        return result

    @fields.setter
    # pylint: disable-next=too-many-branches
    def fields(self, value: list[Union[IEEE754, int, Register64, float]]) -> None:
        """设置字段各个子字段, 保存字段的原始数据

        - 如果 `value` 是 None, 则不做任何操作
        - 更新 `self.bin_value` 和 `self.raw_fields`

        Raises:
            ValueError: 如果字段长度不对

        Args:
            value (list[Union[IEEE754, int, float]]): 子字段列表
                int: 寄存器编号
                IEEE754: 常数浮点数
                Register64: 寄存器对象
                float: 常数浮点数
        """
        if value is None:
            return

        # self.fields 一直存在, 是根据 self.bin_value 计算的
        if len(value) != len(self.fields):
            raise ValueError(f"字段长度不对: {len(value)} != {len(self.fields)}")

        bin_fields = ["" for _ in self.field_format]
        idx = 0
        for i, length in enumerate(self.field_format):
            if length < 0:  # 用 "0" 填充空字段， SMT64位指令段中的reserved
                bin_fields[i] = "0" * abs(length)
                continue

            field = value[idx]
            idx += 1

            if isinstance(field, IEEE754):  # IEEE754 浮点数, length 必须是 32
                if length != 32:
                    raise ValueError(f"数值是 IEEE754, 字段长度必须是 32, {field = }, {length = }")
                bin_value = field.bin_value
            elif isinstance(field, int):  # 整数
                if field < 0:
                    raise ValueError(f"整数 (寄存器编号) {field = } 必须是正数")
                bin_value = IBinary.dec2bin(field, length)
            elif isinstance(field, float):  # 浮点数, length 必须是 32
                if length != 32:
                    raise ValueError(f"数值是 float, 字段长度必须是 32, {field = }, {length = }")
                bin_value = IEEE754(field).bin_value
            elif isinstance(field, Register64):  # 寄存器对象 length 必须是 4
                if length != 4:
                    raise ValueError(f"数值是 Register64, 字段长度必须是 4, {field = }, {length = }")
                bin_value = IBinary.dec2bin(field.index, 4)
            else:
                raise TypeError(f"{field = } ({type(field)}) 类型不对, 只能是int, float, IEEE754 或者 Register64")

            bin_fields[i] = bin_value
        self.bin_value = "".join(bin_fields)
        self.raw_fields = value

    def update_field(self, value: Union[IEEE754, int, str, float], index: int) -> None:
        """更新字段

        Example:
            >>> f = CalField(op_code=OPType.CALCU_REG, fields=[0,10, 0,10, 0,6, 0,10, 0,10, 0,9])
            >>> f.bin_fields
            ['00', '1010', '00', '1010', '0', '0110', '00', '1010', '00', '1010', '0', '1001']
            >>> f.update_field(5, 1)
            >>> f.bin_fields
            ['00', '0101', '00', '1010', '0', '0110', '00', '1010', '00', '1010', '0', '1001']

        Args:
            value (Union[IEEE754, int, str, float]): 值
            index (int): 字段编号
        """
        fields = self.raw_fields.copy()
        fields[index] = value
        self.fields = fields

    @property
    def operand(self) -> str:
        """`self.bin_value` 的别名

        Example:
            >>> f = CalField(op_code=OPType.CALCU_REG, fields=[0,10, 0,10, 0,6, 0,10, 0,10, 0,9])
            >>> f.fields
            [0, 10, 0, 10, 0, 6, 0, 10, 0, 10, 0, 9]
            >>> f.bin_value
            '000000000000000000000010100010100011000101000101001001'
            >>> f.operand
            '000000000000000000000010100010100011000101000101001001'

        Returns:
            str: 二进制操作数
        """
        return self.bin_value


__all__ = [
    "OPType",
    "ALU_OPType", 
    "RS_OPType",
    "RD_OPType",
    "SMT_ASSIGN_I",
    "SMT_RW",
    "SMT_RC",
    "SMT_IC",
    "CalField",
]