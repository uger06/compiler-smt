
from __future__ import annotations
from enum import IntEnum

SMT64_FIELD_FORMAT = {
    # "NOP": [4, 60],
    # "SRAM_LOAD": [4, 60],
    # "SRAM_SAVE": [4, 60],
    # "BUS_LOAD": [4, 60],
    # "BUS_SAVE": [4, 60],
    # "SPIKE_GEN": [4, 60],
    # "ASSIGN_REG": [-24, 8,  6, 6, 5, 5],
    # "ASSIGN_IMM": [ 4, 8, 32, 5, 5],
    "NOP": [54],
    "CALCU_REG": [-20, 2,4, 2,4, 1,4, 2,4, 2,4, 1,4],
    "CALCU_IMM": [32, 2,4, 1,4, 2,4, 1,4],
    # "VSET": [-20, 6, 6, 6, 6, 5, 5],
}
"""SMT 64-bit 指令 field 格式
```python
{
    "NOP": [54],
    "CALCU_REG": [-20, 2,4, 2,4, 1,4, 2,4, 2,4, 1,4],
    "CALCU_IMM": [32, 2,4, 1,4, 2,4, 1,4],
}
```
"""

__all__ = [
    "SMT64_FIELD_FORMAT",
]
