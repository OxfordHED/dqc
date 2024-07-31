from __future__ import annotations

from typing import Callable

import torch

from dqc.utils.datastruct import ValGrad, SpinParam
from dqc.xc.custom_xc import CustomXC
from dqc.xc.base_xc import ZeroXC
from dqc.xc.libxc import get_libxc



def get_linear(a: float, b: float = 0):

    def linear_xc(densinfo):
        if isinstance(densinfo, ValGrad):
            return a * densinfo.value + b
        elif isinstance(densinfo, SpinParam):
            return a * (densinfo.u.value + densinfo.d.value) + b
        else:
            raise NotImplementedError("XC function not implemented for data type of densinfo.")

    return FunctionXC(linear_xc, 1)


def gaussian_distortion(
    mean: float,
    amplitude: float,
    sigma: float = 1.0,
    baseline: str = "lda_x + lda_c_pw"
):


    def gaussian(x: torch.Tensor, mu: float, amp: float, sig: float = 1.0):
        return amp * torch.exp(-0.5 * ((x - mu) / sig)**2) / torch.sqrt(2 * torch.pi * sig)


    def distorted_xc(densinfo):
        baseline_model = sum([get_libxc(comp.strip()) for comp in baseline.split("+")], start=ZeroXC)
        base_xc = baseline_model.get_edensityxc(densinfo)
        if isinstance(densinfo, ValGrad):
            distortion = gaussian(densinfo.value, mean, amplitude, sigma)
        else:
            distortion = gaussian(densinfo.u.value + densinfo.d.value, mean, amplitude, sigma)
        return distortion * base_xc

    return FunctionXC(distorted_xc, 1)

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
