from __future__ import annotations
from itertools import product

import pytest
import numpy as np
import torch
from pyscf import dft, gto

from dqc.system.mol import Mol
from dqc.qccalc.ks import KS
from dqc.utils.periodictable import periodic_table_atomz
from dqc.utils.datastruct import SpinParam

atom_moldescs = [f"{k} 1. 0. 0." for k, v in periodic_table_atomz.items() if v <= 36]

grids = ["sg2", "sg3"] + list(range(10))

@pytest.mark.parametrize(
    "moldesc, grid",
    product(atom_moldescs, grids)
)
def test_dqc_density(moldesc: str, grid: str | int):

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

    def get_total_aodm(qc_calc: object) -> torch.Tensor:
        aodm = qc_calc.aodm()
        return aodm if not isinstance(aodm, SpinParam) else (aodm.u + aodm.d)

    def get_dens(qc_calc: object, r_grid: torch.Tensor) -> torch.Tensor:
        aodmt = get_total_aodm(qc_calc)
        return qc_calc.get_system().get_hamiltonian().aodm2dens(aodmt, r_grid)

    spin = get_spin_0_or_1(moldesc)

    mol_dqc = Mol(moldesc, basis="pc-2", grid=grid, spin=spin)
    mol_dqc.setup_grid()
    dqc_grid = mol_dqc.get_grid().get_rgrid()

    dqc_ks = KS(mol_dqc, "lda_x + lda_c_pw")
    dqc_ks.run()

    dqc_density = get_dens(dqc_ks, dqc_grid)

    pyscf_mol = gto.M(atom=moldesc, basis="pc-2", spin=spin, unit="B")
    pyscf_dft = dft.KS(pyscf_mol, "lda_x,lda_c_pw")
    pyscf_dft.kernel()
    pyscf_dm = pyscf_dft.make_rdm1()
    pyscf_ao = dft.numint.eval_ao(pyscf_mol, dqc_grid.numpy())
    pyscf_density = dft.numint.eval_rho(pyscf_mol, pyscf_ao, pyscf_dm)

    assert np.allclose(pyscf_density, dqc_density, rtol=1e-2)
