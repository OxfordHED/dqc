from itertools import product

import torch
import numpy as np
import pytest
from pyscf import gto, dft

from dqc.grid.radial_grid import RadialGrid
from dqc.grid.lebedev_grid import LebedevGrid
from dqc.grid.multiatoms_grid import BeckeGrid, PBCBeckeGrid
from dqc.grid.factory import get_predefined_grid
from dqc.hamilton.intor.lattice import Lattice
from dqc.utils.periodictable import periodic_table_atomz
from dqc.system.mol import Mol

rgrid_combinations = [
    ("chebyshev", "logm3"),
    ("uniform", "de2"),
    ("chebyshev2", "treutlerm4"),
]

# Check all atoms
atom_moldescs = [f"{k} 1. 0. 0." for k, v in periodic_table_atomz.items() if v <= 36]

@pytest.mark.parametrize(
    "grid_integrator,grid_transform",
    rgrid_combinations
)
def test_radial_grid_dvol(grid_integrator, grid_transform):
    ngrid = 40
    dtype = torch.float64
    radgrid = RadialGrid(ngrid, grid_integrator, grid_transform, dtype=dtype)

    dvol = radgrid.get_dvolume()  # (ngrid,)
    rgrid = radgrid.get_rgrid()  # (ngrid, ndim)
    r = rgrid[:, 0]

    # test gaussian integration
    fcn = torch.exp(-r * r * 0.5)  # (ngrid,)
    int1 = (fcn * dvol).sum()
    val1 = 2 * np.sqrt(2 * np.pi) * np.pi
    assert torch.allclose(int1, int1 * 0 + val1)

@pytest.mark.parametrize(
    "rgrid_integrator,rgrid_transform",
    rgrid_combinations
)
def test_lebedev_grid_dvol(rgrid_integrator, rgrid_transform):
    dtype = torch.float64
    nr = 40
    prec = 7
    radgrid = RadialGrid(nr, rgrid_integrator, rgrid_transform, dtype=dtype)
    sphgrid = LebedevGrid(radgrid, prec=prec)

    dvol = sphgrid.get_dvolume()  # (ngrid,)
    rgrid = sphgrid.get_rgrid()  # (ngrid, ndim)
    x = rgrid[:, 0]
    y = rgrid[:, 1]
    z = rgrid[:, 2]

    # test gaussian integration
    fcn = torch.exp(-(x * x + y * y + z * z) * 0.5)
    int1 = (fcn * dvol).sum()
    val1 = 2 * np.sqrt(2 * np.pi) * np.pi
    assert torch.allclose(int1, int1 * 0 + val1)

@pytest.mark.parametrize(
    "grid_inp",
    [3, 4, "sg2", "sg3"],
)
def test_predefined_grid_dvol(grid_inp):
    dtype = torch.float64
    atomzs = [1]
    atomposs = torch.zeros((1, 3), dtype=dtype)
    sphgrid = get_predefined_grid(grid_inp, atomzs, atomposs, dtype=dtype)

    dvol = sphgrid.get_dvolume()  # (ngrid,)
    rgrid = sphgrid.get_rgrid()  # (ngrid, ndim)
    x = rgrid[:, 0]
    y = rgrid[:, 1]
    z = rgrid[:, 2]

    # test gaussian integration
    fcn = torch.exp(-(x * x + y * y + z * z) * 0.5)
    int1 = (fcn * dvol).sum()
    val1 = 2 * np.sqrt(2 * np.pi) * np.pi
    assert torch.allclose(int1, int1 * 0 + val1)

@pytest.mark.parametrize(
    "rgrid_integrator,rgrid_transform",
    rgrid_combinations
)
def test_multiatoms_grid_dvol(rgrid_integrator, rgrid_transform):
    dtype = torch.float64
    nr = 40
    prec = 7
    radgrid = RadialGrid(nr, rgrid_integrator, rgrid_transform, dtype=dtype)
    sphgrid = LebedevGrid(radgrid, prec=prec)
    atompos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype)
    natoms = atompos.shape[0]
    grid = BeckeGrid([sphgrid, sphgrid], atompos)

    dvol = grid.get_dvolume()  # (ngrid,)
    rgrid = grid.get_rgrid()  # (ngrid, ndim)
    atompos = atompos.unsqueeze(1)  # (natoms, 1, ndim)

    # test gaussian integration
    fcn = torch.exp(-((rgrid - atompos) ** 2).sum(dim=-1) * 0.5).sum(dim=0)  # (ngrid)
    int1 = (fcn * dvol).sum()
    val1 = 2 * (2 * np.sqrt(2 * np.pi) * np.pi)

    # TODO: rtol is relatively large, maybe inspect the Becke integration grid?
    assert torch.allclose(int1, int1 * 0 + val1, rtol=3e-3)

@pytest.mark.parametrize(
    "moldesc,use_xi",
    product(atom_moldescs, [True, False])
)
def test_grid_size(moldesc: str, use_xi: bool):

    def get_spin_0_or_1(moldesc: str) -> int:
        """A helper function to pick the lowest spin with correct parity.

        Args:
            moldesc: str, the formatted description of the molecule

        Returns:
            int, 0 or 1 calculated as num_electrons % 2
        """
        atom_list = moldesc.split(";")
        element_symbols = [atom.strip().split()[0] for atom in atom_list]
        elem_nums = [periodic_table_atomz[es] for es in element_symbols]
        return sum(elem_nums) % 2

    spin = get_spin_0_or_1(moldesc)

    mol_dqc = Mol(moldesc, basis="pc-2", grid=3, spin=spin, use_xi=use_xi)
    mol_dqc.setup_grid()
    dqc_grid = mol_dqc.get_grid().get_rgrid()

    mol_pyscf = gto.M(atom=moldesc, basis="pc-2", spin=spin, unit="B")
    pyscf_dft = dft.KS(mol_pyscf, "lda")
    pyscf_dft.small_rho_cutoff = 1e-21
    pyscf_dft.grids.alignment = 0
    pyscf_dft.kernel()
    pyscf_grid = pyscf_dft.grids.coords

    if not use_xi:
        assert len(dqc_grid.numpy()) == len(pyscf_grid)
    else:
        with pytest.xfail("Xis not implemented in PySCF"):
            assert len(dqc_grid.numpy()) == len(pyscf_grid)

@pytest.mark.parametrize(
    "rgrid_integrator,rgrid_transform",
    rgrid_combinations
)
def test_pbc_multiatoms_grid_dvol(rgrid_integrator, rgrid_transform):
    dtype = torch.float64
    nr = 40
    prec = 7
    radgrid = RadialGrid(nr, rgrid_integrator, rgrid_transform, dtype=dtype)
    sphgrid = LebedevGrid(radgrid, prec=prec)
    atompos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype)
    natoms = atompos.shape[0]
    lattice = Lattice(torch.eye(3, dtype=dtype) * 3)
    grid = PBCBeckeGrid([sphgrid] * natoms, atompos, lattice=lattice)

    dvol = grid.get_dvolume()  # (ngrid,)
    rgrid = grid.get_rgrid()  # (ngrid, ndim)
    ls = lattice.get_lattice_ls(rcut=5)  # (nls, ndim)
    atomposs = (atompos.unsqueeze(1) + ls).reshape(-1, 1, 3)  # (natoms * nls, ndim)

    # test gaussian integration
    fcn = torch.exp(-((rgrid - atomposs) ** 2).sum(dim=-1)).sum(dim=0)  # (ngrid)
    # fcn = rgrid[:, 0] * 0 + 1
    int1 = (fcn * dvol).sum()
    val1 = int1 * 0 + 2 * np.pi ** 1.5  # analytical function

    # TODO: rtol is relatively large, maybe inspect the Becke integration grid?
    assert torch.allclose(int1, int1 * 0 + val1, rtol=1e-2)
