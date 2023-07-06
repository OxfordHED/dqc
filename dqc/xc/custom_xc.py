from abc import abstractmethod, abstractproperty
from typing import Union, List
import torch
from dqc.api.getxc import get_xc
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
    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        pass

    def getparamnames(self, methodname: str = "", prefix: str = "") -> List[str]:
        if methodname == "get_edensityxc":
            pfix = prefix if not prefix.endswith(".") else prefix[:-1]
            names = [name for (name, param) in self.named_parameters(prefix=pfix)]
            return names
        else:
            return super().getparamnames(methodname, prefix=prefix)


class BasicCustomXC(CustomXC):
    """basic xc model joining components for x and c"""

    def __init__(self, models: list[str]):
        assert all(
            [m[: m.find("_")] == models[0][: models[0].find("_")] for m in models]
        ), "base models must belong to same family"
        super().__init__()
        self.models = [get_xc(m) for m in models]
        assert all([m.family is not None and m.family >= 0 for m in self.models])

    @property
    def family(self) -> int:
        return self.models[0].family

    def get_edensityxc(self, densinfo) -> torch.Tensor:
        return torch.stack([m.get_edensityxc(densinfo) for m in self.models]).sum(0)
