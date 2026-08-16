"""Regression tests for the broken-symmetry UKS initial guess.

run(break_symmetry=angle) seeds an unrestricted calculation with a spin-broken
density (HOMO-LUMO mixing) so it can reach the spin-polarized solution that is
the physically-correct dissociation limit -- which UKS cannot find from the
default spin-symmetric guess."""
import pytest
import torch

from dqc.qccalc.ks import KS
from dqc.system.mol import Mol
from dqc.xc.custom_xc import CustomXC
from dqc.utils.safeops import safepow
from dqc.utils.datastruct import ValGrad

dtype = torch.float64
XC = "lda_x + lda_c_pw"


def _h2(dist, basis="3-21G", grid=3):
    pos = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) * dist
    mol = Mol(([1, 1], pos), basis=basis, dtype=dtype, grid=grid)
    mol.setup_grid()
    return mol


def _spin_break(qc):
    dm = qc.aodm()
    return float((dm.u - dm.d).abs().sum())


class _LearnLDA(CustomXC):
    def __init__(self, a=-0.7):
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
def test_broken_symmetry_lower_at_dissociation():
    # past the Coulson-Fischer point the spin-broken UKS solution is lower than
    # the restricted one and is genuinely spin-polarized.
    e_r = float(KS(_h2(5.0), xc=XC, restricted=True).run().energy())
    qc = KS(_h2(5.0), xc=XC, restricted=False).run(break_symmetry=0.785)
    assert qc.is_converged()
    assert float(qc.energy()) < e_r - 1e-3
    assert _spin_break(qc) > 0.5


@pytest.mark.regression
def test_broken_symmetry_collapses_near_equilibrium():
    # near equilibrium there is no broken solution -> it collapses to closed shell
    e_r = float(KS(_h2(1.4), xc=XC, restricted=True).run().energy())
    qc = KS(_h2(1.4), xc=XC, restricted=False).run(break_symmetry=0.785)
    assert qc.is_converged()
    assert abs(float(qc.energy()) - e_r) < 1e-5
    assert _spin_break(qc) < 1e-3


@pytest.mark.regression
def test_break_symmetry_noop_when_restricted():
    # break_symmetry has no spin channels to act on in a restricted calculation
    e0 = float(KS(_h2(5.0), xc=XC, restricted=True).run().energy())
    e1 = float(KS(_h2(5.0), xc=XC, restricted=True).run(break_symmetry=0.785).energy())
    assert e0 == e1


@pytest.mark.regression
def test_broken_symmetry_gradient_matches_finite_difference():
    def energy(a):
        xc = _LearnLDA(a)
        qc = KS(_h2(5.0), xc=xc, restricted=False).run(
            break_symmetry=0.785)
        assert qc.is_converged()
        return qc.energy(), xc

    e, xc = energy(-0.7)
    e.backward()
    g_auto = float(xc.a.grad)
    eps = 1e-4
    with torch.no_grad():
        ep = float(energy(-0.7 + eps)[0].detach())
        em = float(energy(-0.7 - eps)[0].detach())
    g_fd = (ep - em) / (2 * eps)
    assert abs(g_auto - g_fd) / abs(g_fd) < 1e-4
