from collections import namedtuple
import torch
import pytest
from ddft.basissets.cgtobasis import loadbasis
from ddft.hamiltons.libcintwrapper import LibcintWrapper
from ddft.basissets.cgtobasis import AtomCGTOBasis

AtomEnv = namedtuple("AtomEnv", ["poss", "basis", "rgrid", "atomzs"])

def get_atom_env(dtype, rgrid=False):
    pos1 = torch.tensor([0.0, 0.0,  0.8], dtype=dtype, requires_grad=True)
    pos2 = torch.tensor([0.0, 0.0, -0.8], dtype=dtype, requires_grad=True)
    poss = [pos1, pos2]
    atomzs = [1, 1]
    basis = "3-21G"

    # set the grid
    n = 3
    z = torch.linspace(-5, 5, n, dtype=dtype)
    zeros = torch.zeros(n, dtype=dtype)
    rgrid = torch.cat((zeros[None, :], zeros[None, :], z[None, :]), dim=0).T.contiguous().to(dtype)

    return AtomEnv(
        poss = poss,
        basis = basis,
        rgrid = rgrid,
        atomzs = atomzs
    )

@pytest.mark.parametrize(
    "int_type",
    ["overlap", "kinetic"]
)
def test_integral_grad(int_type):
    dtype = torch.double

    atomenv = get_atom_env(dtype)
    pos1 = atomenv.poss[0]
    pos2 = atomenv.poss[1]
    allbases = [
        loadbasis("%d:%s" % (atomz, atomenv.basis), dtype=dtype, requires_grad=False) \
        for atomz in atomenv.atomzs
    ]

    def get_int1e(pos1, pos2, name):
        atombasis1 = AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=allbases[0], pos=pos1)
        atombasis2 = AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=allbases[1], pos=pos2)
        env = LibcintWrapper([atombasis1, atombasis2], spherical=True)
        if name == "overlap":
            return env.overlap()
        elif name == "kinetic":
            return env.kinetic()
        elif name == "nuclattr":
            return env.nuclattr()
        elif name == "elrep":
            return env.elrep()
        else:
            raise RuntimeError()

    # integrals gradcheck
    torch.autograd.gradcheck(get_int1e, (pos1, pos2, int_type))
    torch.autograd.gradgradcheck(get_int1e, (pos1, pos2, int_type))

# TODO: complete this
# @pytest.mark.parametrize(
#     "eval_type",
#     ["", "grad"]
# )
# def test_eval_vs_pyscf(eval_type):
#     pass

@pytest.mark.parametrize(
    "eval_type",
    ["", "grad", "lapl"]
)
def test_eval_grad(eval_type):
    dtype = torch.double

    atomenv = get_atom_env(dtype, rgrid=True)
    pos1 = atomenv.poss[0]
    pos2 = atomenv.poss[1]
    allbases = [
        loadbasis("%d:%s" % (atomz, atomenv.basis), dtype=dtype, requires_grad=False) \
        for atomz in atomenv.atomzs
    ]
    rgrid = atomenv.rgrid

    def evalgto(pos1, pos2, rgrid, name):
        atombasis1 = AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=allbases[0], pos=pos1)
        atombasis2 = AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=allbases[1], pos=pos2)
        env = LibcintWrapper([atombasis1, atombasis2], spherical=True)
        if name == "":
            return env.eval_gto(rgrid)
        elif name == "grad":
            return env.eval_gradgto(rgrid)
        elif name == "lapl":
            return env.eval_laplgto(rgrid)
        else:
            raise RuntimeError("Unknown name: %s" % name)

    # evals gradcheck
    torch.autograd.gradcheck(evalgto, (pos1, pos2, rgrid, eval_type))
    torch.autograd.gradgradcheck(evalgto, (pos1, pos2, rgrid, eval_type))