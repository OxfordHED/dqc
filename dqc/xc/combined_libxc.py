import torch
from dqc.xc.custom_xc import CustomXC, ZeroXC
from dqc.xc.libxc import get_libxc


class BasicCustomXC(CustomXC):
    """basic xc model joining components for x and c"""

    def __init__(self, models: list[str]):
        if not models:
            self.models = [ZeroXC()]
        else:
            assert all(
                [m[: m.find("_")] == models[0][: models[0].find("_")] for m in models]
            ), "base models must belong to same family"
            super().__init__()
            self.models = [get_libxc(m) for m in models]
            assert all([m.family is not None and m.family >= 0 for m in self.models])

    @property
    def family(self) -> int:
        return self.models[0].family

    def get_edensityxc(self, densinfo) -> torch.Tensor:
        return torch.stack([m.get_edensityxc(densinfo) for m in self.models]).sum(0)
