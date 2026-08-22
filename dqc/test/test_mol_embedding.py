import pytest
import torch

from dqc.system.mol import Mol
from dqc.utils.datastruct import ValGrad, SpinParam

# tests for the per-grid-point node features handed to graph (nonlocal) xc models

dtype = torch.float64
moldesc = "H 1.0 0.0 0.0; H -1.0 0.0 0.0"


def _mol():
    m = Mol(moldesc, basis="3-21G", dtype=dtype, grid=1)
    m.setup_grid()
    return m


def _densinfo(nr, polarized=True, grad=True, seed=0):
    torch.manual_seed(seed)

    def one():
        val = torch.rand((nr,), dtype=dtype) + 0.1
        g = torch.rand((3, nr), dtype=dtype) - 0.5 if grad else None
        return ValGrad(value=val, grad=g)

    return SpinParam(u=one(), d=one()) if polarized else one()


def test_lda_embedding_layout_unchanged():
    # regression guard: the default block must stay (n, zeta, r, Z)
    mol = _mol()
    embed = mol.get_embedding()
    nr = embed._radial_dists.shape[0]
    densinfo = _densinfo(nr)

    out = embed.apply(densinfo)
    assert embed.n_density_features == 2
    assert out.shape == (nr, 4)

    dens = densinfo.u.value + densinfo.d.value
    assert torch.allclose(out[:, 0], dens)
    assert torch.allclose(out[:, 1], (densinfo.u.value - densinfo.d.value) / dens)
    assert torch.allclose(out[:, 2], embed._radial_dists)
    assert torch.allclose(out[:, 3], embed._atom_zs.to(dtype))


def test_gga_embedding_polarized():
    mol = _mol()
    embed = mol.get_embedding(gga=True)
    nr = embed._radial_dists.shape[0]
    densinfo = _densinfo(nr)

    out = embed.apply(densinfo)
    assert embed.n_density_features == 4
    assert out.shape == (nr, 6)

    assert torch.allclose(out[:, 0], densinfo.u.value)
    assert torch.allclose(out[:, 1], densinfo.d.value)
    assert torch.allclose(out[:, 2], densinfo.u.grad.norm(dim=-2))
    assert torch.allclose(out[:, 3], densinfo.d.grad.norm(dim=-2))
    # the grid descriptors keep their trailing position
    assert torch.allclose(out[:, 4], embed._radial_dists)
    assert torch.allclose(out[:, 5], embed._atom_zs.to(dtype))


def test_gga_embedding_unpolarized_splits_evenly():
    # value/grad carry the total density in the unpolarized case
    mol = _mol()
    embed = mol.get_embedding(gga=True)
    nr = embed._radial_dists.shape[0]
    densinfo = _densinfo(nr, polarized=False)

    out = embed.apply(densinfo)
    assert torch.allclose(out[:, 0], 0.5 * densinfo.value)
    assert torch.allclose(out[:, 1], 0.5 * densinfo.value)
    assert torch.allclose(out[:, 2], 0.5 * densinfo.grad.norm(dim=-2))
    assert torch.allclose(out[:, 3], 0.5 * densinfo.grad.norm(dim=-2))


def test_gga_embedding_is_rotationally_invariant():
    # a functional of the raw gradient vectors would not be
    mol = _mol()
    embed = mol.get_embedding(gga=True)
    nr = embed._radial_dists.shape[0]
    densinfo = _densinfo(nr)

    theta = torch.tensor(0.7, dtype=dtype)
    c, s = torch.cos(theta), torch.sin(theta)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    rotated = SpinParam(
        u=ValGrad(value=densinfo.u.value, grad=rot @ densinfo.u.grad),
        d=ValGrad(value=densinfo.d.value, grad=rot @ densinfo.d.grad),
    )

    assert torch.allclose(embed.apply(densinfo), embed.apply(rotated))


def test_gga_embedding_is_differentiable_at_zero_gradient():
    # vxc is obtained by differentiating through these features, so a bare
    # sqrt(sigma) would hand back nan wherever the gradient vanishes
    mol = _mol()
    embed = mol.get_embedding(gga=True)
    nr = embed._radial_dists.shape[0]

    val = torch.rand((nr,), dtype=dtype) + 0.1
    grad = torch.zeros((3, nr), dtype=dtype, requires_grad=True)
    densinfo = SpinParam(
        u=ValGrad(value=val, grad=grad), d=ValGrad(value=val, grad=grad)
    )

    out = embed.apply(densinfo)
    assert torch.all(torch.isfinite(out))
    out.sum().backward()
    assert torch.all(torch.isfinite(grad.grad))


def test_gga_embedding_grad_norm_derivative():
    mol = _mol()
    embed = mol.get_embedding(gga=True)
    nr = embed._radial_dists.shape[0]

    val = torch.rand((nr,), dtype=dtype) + 0.1
    g = torch.rand((3, nr), dtype=dtype) + 0.1
    g.requires_grad_(True)
    densinfo = SpinParam(
        u=ValGrad(value=val, grad=g), d=ValGrad(value=val, grad=g.detach())
    )

    embed.apply(densinfo)[:, 2].sum().backward()
    # d|g| / dg = g / |g|
    assert torch.allclose(g.grad, g.detach() / g.detach().norm(dim=-2))


def test_gga_embedding_requires_gradients():
    mol = _mol()
    embed = mol.get_embedding(gga=True)
    nr = embed._radial_dists.shape[0]
    densinfo = _densinfo(nr, grad=False)

    with pytest.raises(RuntimeError, match="family >= 2"):
        embed.apply(densinfo)


def test_get_embedding_toggles_gga_but_not_coords():
    mol = _mol()
    embed = mol.get_embedding()
    assert embed.n_density_features == 2

    # gga only changes what apply() computes, so it may be flipped in place
    assert mol.get_embedding(gga=True) is embed
    assert embed.n_density_features == 4
    assert mol.get_embedding(gga=False) is embed
    assert embed.n_density_features == 2

    # the coordinate layout is fixed at build time and must not be faked
    with pytest.raises(RuntimeError, match="append_raw_coords"):
        mol.get_embedding(append_raw_coords=True)


def test_gga_embedding_with_raw_coords():
    mol = _mol()
    embed = mol.get_embedding(append_raw_coords=True, gga=True)
    nr = embed._coordinates.shape[0]
    densinfo = _densinfo(nr)

    out = embed.apply(densinfo)
    assert out.shape == (nr, 7)  # 4 density + 3 coordinates
    assert torch.allclose(out[:, 4:], embed._coordinates)
