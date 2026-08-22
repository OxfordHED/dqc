import pytest
import torch

from dqc.system.mol import Mol
from dqc.utils.datastruct import ValGrad, SpinParam

# tests for the per-grid-point node features handed to graph (nonlocal) xc models

dtype = torch.float64
moldesc = "H 1.0 0.0 0.0; H -1.0 0.0 0.0"
NR = 40


def _mol():
    m = Mol(moldesc, basis="3-21G", dtype=dtype, grid=1)
    m.setup_grid()
    return m


def _densinfo(nr=NR, polarized=True, grad=True, seed=0):
    torch.manual_seed(seed)

    def one():
        val = torch.rand((nr,), dtype=dtype) + 0.1
        g = torch.rand((3, nr), dtype=dtype) - 0.5 if grad else None
        return ValGrad(value=val, grad=g)

    return SpinParam(u=one(), d=one()) if polarized else one()


def test_lda_embedding_is_density_only():
    # the grid descriptors (radial_dists, atom_zs) are gone: the features are
    # the xc ingredients and nothing else
    embed = _mol().get_embedding()
    densinfo = _densinfo()

    out = embed.apply(densinfo)
    assert embed.n_density_features == 2
    assert out.shape == (NR, 2)

    dens = densinfo.u.value + densinfo.d.value
    assert torch.allclose(out[:, 0], dens)
    assert torch.allclose(out[:, 1], (densinfo.u.value - densinfo.d.value) / dens)


def test_gga_embedding_polarized():
    embed = _mol().get_embedding(gga=True)
    densinfo = _densinfo()

    out = embed.apply(densinfo)
    assert embed.n_density_features == 5
    assert out.shape == (NR, 5)

    gu, gd = densinfo.u.grad, densinfo.d.grad
    assert torch.allclose(out[:, 0], densinfo.u.value)
    assert torch.allclose(out[:, 1], densinfo.d.value)
    assert torch.allclose(out[:, 2], gu.norm(dim=-2))
    assert torch.allclose(out[:, 3], gd.norm(dim=-2))
    assert torch.allclose(out[:, 4], (gu * gd).sum(dim=-2))


def test_cross_term_spans_libxc_sigmas():
    # (|grad n_u|, |grad n_d|, sigma_ud) must carry the same information as
    # libxc's (sigma_uu, sigma_ud, sigma_dd)
    embed = _mol().get_embedding(gga=True)
    densinfo = _densinfo()
    out = embed.apply(densinfo)

    gu, gd = densinfo.u.grad, densinfo.d.grad
    assert torch.allclose(out[:, 2] ** 2, (gu * gu).sum(dim=-2))  # sigma_uu
    assert torch.allclose(out[:, 3] ** 2, (gd * gd).sum(dim=-2))  # sigma_dd

    # and the total-density gradient is recoverable, which spin-polarized
    # correlation needs: |grad n|^2 = sigma_uu + 2 sigma_ud + sigma_dd
    total = out[:, 2] ** 2 + 2 * out[:, 4] + out[:, 3] ** 2
    assert torch.allclose(total, ((gu + gd) * (gu + gd)).sum(dim=-2))


def test_cross_term_sees_the_angle():
    # this is the degree of freedom the magnitudes alone cannot represent:
    # parallel and antiparallel spin gradients of equal length must differ
    embed = _mol().get_embedding(gga=True)
    torch.manual_seed(0)
    val = torch.rand((NR,), dtype=dtype) + 0.1
    g = torch.rand((3, NR), dtype=dtype) + 0.1

    para = SpinParam(u=ValGrad(value=val, grad=g), d=ValGrad(value=val, grad=g))
    anti = SpinParam(u=ValGrad(value=val, grad=g), d=ValGrad(value=val, grad=-g))

    out_p, out_a = embed.apply(para), embed.apply(anti)
    # densities and both magnitudes agree...
    assert torch.allclose(out_p[:, :4], out_a[:, :4])
    # ...only the cross term distinguishes them, and it flips sign
    assert torch.allclose(out_p[:, 4], -out_a[:, 4])
    assert not torch.allclose(out_p[:, 4], out_a[:, 4])


def test_gga_embedding_unpolarized_splits_evenly():
    embed = _mol().get_embedding(gga=True)
    densinfo = _densinfo(polarized=False)

    out = embed.apply(densinfo)
    half_grad = 0.5 * densinfo.grad
    assert torch.allclose(out[:, 0], 0.5 * densinfo.value)
    assert torch.allclose(out[:, 1], 0.5 * densinfo.value)
    assert torch.allclose(out[:, 2], half_grad.norm(dim=-2))
    assert torch.allclose(out[:, 3], half_grad.norm(dim=-2))
    assert torch.allclose(out[:, 4], (half_grad * half_grad).sum(dim=-2))


def test_gga_embedding_is_rotationally_invariant():
    embed = _mol().get_embedding(gga=True)
    densinfo = _densinfo()

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
    embed = _mol().get_embedding(gga=True)
    val = torch.rand((NR,), dtype=dtype) + 0.1
    grad = torch.zeros((3, NR), dtype=dtype, requires_grad=True)
    densinfo = SpinParam(
        u=ValGrad(value=val, grad=grad), d=ValGrad(value=val, grad=grad)
    )

    out = embed.apply(densinfo)
    assert torch.all(torch.isfinite(out))
    out.sum().backward()
    assert torch.all(torch.isfinite(grad.grad))


def test_gga_embedding_grad_norm_derivative():
    embed = _mol().get_embedding(gga=True)
    val = torch.rand((NR,), dtype=dtype) + 0.1
    g = torch.rand((3, NR), dtype=dtype) + 0.1
    g.requires_grad_(True)
    densinfo = SpinParam(
        u=ValGrad(value=val, grad=g), d=ValGrad(value=val, grad=g.detach())
    )

    embed.apply(densinfo)[:, 2].sum().backward()
    # d|g| / dg = g / |g|
    assert torch.allclose(g.grad, g.detach() / g.detach().norm(dim=-2))


def test_gga_embedding_requires_gradients():
    embed = _mol().get_embedding(gga=True)
    with pytest.raises(RuntimeError, match="family >= 2"):
        embed.apply(_densinfo(grad=False))


def test_get_embedding_toggles_gga_but_not_coords():
    mol = _mol()
    embed = mol.get_embedding()
    assert embed.n_density_features == 2

    # gga only changes what apply() computes, so it may be flipped in place
    assert mol.get_embedding(gga=True) is embed
    assert embed.n_density_features == 5
    assert mol.get_embedding(gga=False) is embed
    assert embed.n_density_features == 2

    # the coordinate layout is fixed at build time and must not be faked
    with pytest.raises(RuntimeError, match="append_raw_coords"):
        mol.get_embedding(append_raw_coords=True)


def test_raw_coords_are_appended_after_the_density_block():
    mol = _mol()
    embed = mol.get_embedding(append_raw_coords=True, gga=True)
    nr = embed._coordinates.shape[0]

    out = embed.apply(_densinfo(nr=nr))
    assert out.shape == (nr, 8)  # 5 density + 3 coordinates
    assert torch.allclose(out[:, embed.n_density_features :], embed._coordinates)
