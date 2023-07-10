from __future__ import annotations

from abc import abstractmethod, abstractproperty
from typing import List
import torch
from dqc.xc.base_xc import BaseXC
from dqc.utils.datastruct import ValGrad, SpinParam

class CustomXC(BaseXC, torch.nn.Module):
    """
    Base class of custom xc functional.
    """
    @abstractproperty
    def family(self) -> int:
        pass

    @abstractmethod
    def get_edensityxc(self, densinfo: ValGrad | SpinParam[ValGrad]) -> torch.Tensor:
        pass

    def getparamnames(self, methodname: str = "", prefix: str = "") -> List[str]:
        if methodname == "get_edensityxc":
            pfix = prefix if not prefix.endswith(".") else prefix[:-1]
            names = [name for (name, param) in self.named_parameters(prefix=pfix)]
            return names
        else:
            return super().getparamnames(methodname, prefix=prefix)


class ZeroXC(CustomXC):

    family = 0

    def get_edensityxc(self, densinfo: ValGrad | SpinParam[ValGrad]) -> torch.Tensor:
        if isinstance(densinfo, SpinParam):
            val_grad = densinfo.u
        else:
            val_grad = densinfo

        shape = val_grad.value.shape
        return torch.zeros(shape)

    def get_vxc(self, densinfo: ValGrad | SpinParam[ValGrad]) -> torch.Tensor:
        with self._enable_grad_densinfo:  # Unsure if this is required here
            edensityxc = self.get_edensityxc(densinfo)  # all zeros
            if isinstance(densinfo, SpinParam):
                # all zeros
                return SpinParam(u=ValGrad(value=edensityxc), d=ValGrad(value=edensityxc))
            else:
                # all zeros
                return ValGrad(value=edensityxc)

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "get_vxc":
            return self.getparamnames("get_edensityxc", prefix=prefix)
        else:
            raise KeyError("Unknown methodname: %s" % methodname)