from __future__ import annotations

from typing import Callable

import torch

from dqc.utils.datastruct import ValGrad, SpinParam
from dqc.xc.custom_xc import CustomXC



def get_linear(a: float, b: float = 0):

    def linear_xc(densinfo):
        if isinstance(densinfo, ValGrad):
            return a * densinfo.value + b
        elif isinstance(densinfo, SpinParam):
            return a * (densinfo.u.value + densinfo.d.value) + b
        else:
            raise NotImplementedError("XC function not implemented for data type of densinfo.")

    return FunctionXC(linear_xc, 1)



class FunctionXC(CustomXC):

    def __init__(self, function: Callable, family: int):
        super().__init__()
        self._family = family
        self._function = function

    @property
    def family(self) -> int:
        return self._family

    def get_edensityxc(self, densinfo: ValGrad | SpinParam[ValGrad]) -> torch.Tensor:
        return self._function(densinfo)
