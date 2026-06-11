"""Regression tests for Fermi (finite-temperature) smearing.

Smearing gives the SCF fractional occupations, which stabilise near-degenerate /
dissociating systems where rigid integer aufbau has no stable fixed point. These
tests pin: (a) it converges a geometry that fails without it, (b) it does not
perturb a large-gap energy, and (c) the learning gradient through the smeared
SCF is correct where it was silently wrong without smearing."""
import pytest
import torch

from dqc.qccalc.ks import KS
from dqc.system.mol import Mol
from dqc.xc.custom_xc import CustomXC
from dqc.utils.safeops import safepow
from dqc.utils.datastruct import ValGrad

dtype = torch.float64


def _h2(dist, basis="3-21G", grid=3):
    pos = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) * dist
    mol = Mol(([1, 1], pos), basis=basis, dtype=dtype, grid=grid)
    mol.setup_grid()
    return mol


class _LearnLDA(CustomXC):
    def __init__(self, a=-0.7385587663820223):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a, dtype=dtype))

    @property
    def family(self):
        return 1

    def get_edensityxc(self, densinfo):
        if isinstance(densinfo, ValGrad):
            return self.a * safepow(densinfo.value.abs(), 4.0 / 3.0)
        return 0.5 * (self.get_edensityxc(densinfo.u * 2)
                      + self.get_edensityxc(densinfo.d * 2))


@pytest.mark.regression
def test_smearing_rescues_dissociation():
    # stretched H2: restricted SCF does not converge without smearing ...
    qc0 = KS(_h2(12.0), xc="lda_x + lda_c_pw", restricted=True).run(diagnose=True)
    assert qc0.is_converged() is False
    # ... but does with Fermi smearing.
    qc1 = KS(_h2(12.0), xc="lda_x + lda_c_pw", restricted=True, smearing=0.02).run(diagnose=True)
    assert qc1.is_converged() is True
    assert qc1.get_scf_diagnostics()["residual"] < 1e-5


@pytest.mark.regression
def test_smearing_preserves_large_gap_energy():
    # for a large-gap system the occupations are ~integer, so light smearing
    # must not change the energy.
    e_no = float(KS(_h2(1.4), xc="lda_x + lda_c_pw", restricted=True).run().energy())
    e_sm = float(KS(_h2(1.4), xc="lda_x + lda_c_pw", restricted=True,
                    smearing=0.01).run().energy())
    assert abs(e_no - e_sm) < 1e-5


@pytest.mark.regression
def test_smearing_entropy_and_free_energy():
    kT = 0.02
    # large gap: entropy ~ 0
    qc_big = KS(_h2(1.4), xc="lda_x + lda_c_pw", restricted=True, smearing=kT).run()
    assert float(qc_big.entropy()) < 1e-2
    # full dissociation: each of the two near-degenerate orbitals is half filled,
    # so S ~ 4 ln 2 ~ 2.77
    qc = KS(_h2(12.0), xc="lda_x + lda_c_pw", restricted=True, smearing=kT).run()
    S = float(qc.entropy())
    assert 2.0 < S < 3.0
    E = float(qc.energy())
    assert abs(float(qc.free_energy()) - (E - kT * S)) < 1e-9
    assert abs(float(qc.energy0()) - (E - 0.5 * kT * S)) < 1e-9
    # no smearing: entropy 0 and the three energies coincide
    qc0 = KS(_h2(1.4), xc="lda_x + lda_c_pw", restricted=True).run()
    assert float(qc0.entropy()) == 0.0
    assert float(qc0.free_energy()) == float(qc0.energy())
    assert float(qc0.energy0()) == float(qc0.energy())


@pytest.mark.regression
def test_smearing_gradient_matches_finite_difference():
    # the learning gradient through the smeared SCF is correct at a geometry
    # that would not converge (and give a wrong gradient) without smearing.
    kT = 0.02

    def energy(a):
        xc = _LearnLDA(a)
        qc = KS(_h2(18.0), xc=xc, restricted=True, smearing=kT).run(diagnose=True, conv_tol=1e-6)
        assert qc.is_converged()
        return qc.energy(), xc

    a0 = -0.7
    e, xc = energy(a0)
    e.backward()
    g_auto = float(xc.a.grad)
    eps = 1e-4
    with torch.no_grad():
        ep = float(energy(a0 + eps)[0].detach())
        em = float(energy(a0 - eps)[0].detach())
    g_fd = (ep - em) / (2 * eps)
    assert abs(g_auto - g_fd) / abs(g_fd) < 1e-4
