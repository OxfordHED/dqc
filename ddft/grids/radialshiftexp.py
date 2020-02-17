import torch
import numpy as np
from ddft.grids.base_grid import BaseGrid

class RadialShiftExp(BaseGrid):
    def __init__(self, rmin, rmax, nr, dtype=torch.float, device=torch.device('cpu')):
        logr = torch.linspace(np.log(rmin), np.log(rmax), nr).to(dtype).to(device)
        unshifted_rgrid = torch.exp(logr)
        self._boxshape = (nr,)
        self.rmin = rmin
        self.rs = unshifted_rgrid - self.rmin
        self._rgrid = self.rs.unsqueeze(1) # (nr, 1)
        self.dlogr = logr[1] - logr[0]
        self._dvolume = (self.rs + self.rmin) * 4 * np.pi * self.rs*self.rs * self.dlogr

    def get_dvolume(self):
        return self._dvolume

    def solve_poisson(self, f):
        # f: (nbatch, nr)
        # the expression below is used to make the operator symmetric
        eps = 1e-10
        intgn1 = f * self.rs * self.rs * (self.rs + self.rmin) * self.dlogr
        int1 = torch.cumsum(intgn1, dim=-1)
        intgn2 = int1 / (self.rs * self.rs + eps) * (self.rs + self.rmin) * self.dlogr
        # this form of cumsum is the transpose of torch.cumsum
        int2 = -torch.cumsum(intgn2.flip(dims=[-1]), dim=-1).flip(dims=[-1])
        return int2

    @property
    def rgrid(self):
        return self._rgrid

    @property
    def boxshape(self):
        return self._boxshape