from __future__ import annotations

from contextlib import contextmanager
from abc import abstractmethod, abstractproperty
import torch
import xitorch as xt
from typing import List, Union, overload, Iterator
from dqc.utils.datastruct import ValGrad, SpinParam


class BaseXC(xt.EditableModule):
    """
    XC is class that calculates the components of xc potential and energy
    density given the density.
    """

    @abstractproperty
    def family(self) -> int:
        """
        Returns 1 for LDA, 2 for GGA, and 4 for Meta-GGA.
        """
        pass

    @abstractmethod
    def get_edensityxc(
        self, densinfo: Union[ValGrad, SpinParam[ValGrad]]
    ) -> torch.Tensor:
        """
        Returns the xc energy density (energy per unit volume)
        """
        # densinfo.value & lapl: (*BD, nr)
        # densinfo.grad: (*BD, ndim, nr)
        # return: (*BD, nr)
        pass

    @overload
    def get_vxc(self, densinfo: ValGrad) -> ValGrad: ...

    @overload
    def get_vxc(self, densinfo: SpinParam[ValGrad]) -> SpinParam[ValGrad]: ...

    def get_vxc(self, densinfo, embed=None, graph=None, edge_feats=None):
        """
        Returns the ValGrad for the xc potential given the density info
        for unpolarized case.
        """
        # This is the default implementation of vxc if there is no implementation
        # in the specific class of XC.

        # densinfo.value & lapl: (*BD, nr)
        # densinfo.grad: (*BD, ndim, nr)
        # return:
        # potentialinfo.value & lapl: (*BD, nr)
        # potentialinfo.grad: (*BD, ndim, nr)

        kwargs = {}
        if embed is not None or graph is not None:
            kwargs["embed"] = embed
            kwargs["graph"] = graph
            kwargs["edge_feats"] = edge_feats

        # The vxc that makes the Fock matrix the true derivative of the energy
        # E = sum_r' w_r' e(r') is the weighted column sum
        # sum_r' w_r' d(e(r')) / d(dens(r)), i.e. a VJP with the quadrature
        # weights as the output vector. For a local functional this reduces to
        # w_r * d(e(r)) / d(dens(r)), so weighting here and skipping the weight
        # in the matrix assembly (potinfo.weighted below) is exact for local
        # functionals too — but it is required for nonlocal functionals (e.g.
        # graph models) that couple grid points, where a plain ones-VJP is not
        # the derivative of any energy. Without a grid we fall back to the
        # ones-VJP, which is only correct for local functionals.
        grid = densinfo.grid if isinstance(densinfo, ValGrad) else densinfo.u.grid
        weighted = grid is not None

        # mark the densinfo components as requiring grads
        with self._enable_grad_densinfo(densinfo):
            with torch.enable_grad():
                edensity = self.get_edensityxc(densinfo, **kwargs)  # (*BD, nr)
            if weighted:
                dvolume = grid.get_dvolume().to(edensity.dtype)
                grad_outputs = dvolume.expand_as(edensity).contiguous()
            else:
                grad_outputs = torch.ones_like(edensity)
            grad_enabled = torch.is_grad_enabled()

            if not isinstance(densinfo, ValGrad):  # polarized case
                if self.family == 1:  # LDA
                    params = (densinfo.u.value, densinfo.d.value)
                    dedn_u, dedn_d = torch.autograd.grad(
                        edensity,
                        params,
                        create_graph=grad_enabled,
                        grad_outputs=grad_outputs,
                    )

                    return SpinParam(
                        u=ValGrad(value=dedn_u, weighted=weighted),
                        d=ValGrad(value=dedn_d, weighted=weighted),
                    )

                elif self.family == 2:  # GGA
                    params = (
                        densinfo.u.value,
                        densinfo.d.value,
                        densinfo.u.grad,
                        densinfo.d.grad,
                    )
                    dedn_u, dedn_d, dedg_u, dedg_d = torch.autograd.grad(
                        edensity,
                        params,
                        create_graph=grad_enabled,
                        grad_outputs=grad_outputs,
                    )

                    return SpinParam(
                        u=ValGrad(value=dedn_u, grad=dedg_u, weighted=weighted),
                        d=ValGrad(value=dedn_d, grad=dedg_d, weighted=weighted),
                    )

                elif self.family == 4:
                    params = (
                        densinfo.u.value,
                        densinfo.d.value,
                        densinfo.u.grad,
                        densinfo.d.grad,
                        densinfo.u.lapl,
                        densinfo.d.lapl,
                        densinfo.u.kin,
                        densinfo.d.kin,
                    )
                    dedn_u, dedn_d, dedg_u, dedg_d, dedl_u, dedl_d, dedk_u, dedk_d = (
                        torch.autograd.grad(
                            edensity,
                            params,
                            create_graph=grad_enabled,
                            grad_outputs=grad_outputs,
                            allow_unused=True,
                        )
                    )

                    # mgga might only use one of either lapl or kin, so we need to change the deriv manually to 0s
                    dedl_u = dedl_u if dedl_u is not None else torch.zeros_like(dedn_u)
                    dedk_u = dedk_u if dedk_u is not None else torch.zeros_like(dedn_u)
                    dedl_d = dedl_d if dedl_d is not None else torch.zeros_like(dedn_d)
                    dedk_d = dedk_d if dedk_d is not None else torch.zeros_like(dedn_d)

                    return SpinParam(
                        u=ValGrad(value=dedn_u, grad=dedg_u, lapl=dedl_u,
                                  kin=dedk_u, weighted=weighted),
                        d=ValGrad(value=dedn_d, grad=dedg_d, lapl=dedl_d,
                                  kin=dedk_d, weighted=weighted),
                    )

                else:
                    raise NotImplementedError(
                        "Default polarized vxc for family %s is not implemented"
                        % self.family
                    )

            else:  # unpolarized case
                if self.family == 1:  # LDA
                    (dedn,) = torch.autograd.grad(
                        edensity,
                        densinfo.value,
                        create_graph=grad_enabled,
                        grad_outputs=grad_outputs,
                    )

                    return ValGrad(value=dedn, weighted=weighted)

                elif self.family == 2:  # GGA
                    dedn, dedg = torch.autograd.grad(
                        edensity,
                        (densinfo.value, densinfo.grad),
                        create_graph=grad_enabled,
                        grad_outputs=grad_outputs,
                    )

                    return ValGrad(value=dedn, grad=dedg, weighted=weighted)

                elif self.family == 4:  # MGGA
                    dedn, dedg, dedl, dedk = torch.autograd.grad(
                        edensity,
                        (densinfo.value, densinfo.grad, densinfo.lapl, densinfo.kin),
                        create_graph=grad_enabled,
                        grad_outputs=grad_outputs,
                        allow_unused=True,
                    )

                    # mgga might only use one of either lapl or kin, so we need to change the deriv manually to 0s
                    dedl = dedl if dedl is not None else torch.zeros_like(dedn)
                    dedk = dedk if dedk is not None else torch.zeros_like(dedn)

                    return ValGrad(value=dedn, grad=dedg, lapl=dedl, kin=dedk,
                                   weighted=weighted)

                else:
                    raise NotImplementedError(
                        "Default vxc for family %d is not implemented" % self.family
                    )

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "get_vxc":
            return self.getparamnames("get_edensityxc", prefix=prefix)
        else:
            raise KeyError("Unknown methodname: %s" % methodname)

    @contextmanager
    def _enable_grad_densinfo(
        self, densinfo: Union[ValGrad, SpinParam[ValGrad]]
    ) -> Iterator:
        # set the context where some elements (depends on xc family) in densinfo requires grad

        def _get_set_grad(vars: List[torch.Tensor]) -> List[bool]:
            # set the vars to require grad and returns the previous state of the vars
            reqgrads = []
            for var in vars:
                reqgrads.append(var.requires_grad)
                var.requires_grad_()
            return reqgrads

        def _restore_grad(reqgrads: List[bool], vars: List[torch.Tensor]) -> None:
            # restore the state of requiring grad based on reqgrads list
            # all vars before this function requires grad
            for reqgrad, var in zip(reqgrads, vars):
                if not reqgrad:
                    var.requires_grad_(False)

        # getting which parameters should require grad
        if not isinstance(densinfo, ValGrad):  # a spinparam
            params = [densinfo.u.value, densinfo.d.value]
            if self.family >= 2:  # GGA
                assert densinfo.u.grad is not None
                assert densinfo.d.grad is not None
                params.extend([densinfo.u.grad, densinfo.d.grad])
            if self.family >= 3:  # MGGA
                assert densinfo.u.lapl is not None
                assert densinfo.d.lapl is not None
                assert densinfo.u.kin is not None
                assert densinfo.d.kin is not None
                params.extend(
                    [densinfo.u.lapl, densinfo.d.lapl, densinfo.u.kin, densinfo.d.kin]
                )
        else:
            params = [densinfo.value]
            if self.family >= 2:
                assert densinfo.grad is not None
                params.append(densinfo.grad)
            if self.family >= 3:
                assert densinfo.lapl is not None
                assert densinfo.kin is not None
                params.extend([densinfo.lapl, densinfo.kin])

        try:
            # set the params to require grad
            reqgrads = _get_set_grad(params)
            yield
        finally:
            _restore_grad(reqgrads, params)

    # special operations
    def __add__(self, other):
        return AddBaseXC(self, other)

    def __mul__(self, other: Union[float, int, torch.Tensor]):
        if isinstance(other, float) or isinstance(other, int):
            return MulBaseXC(self, float(other))
        elif isinstance(other, torch.Tensor):
            return MulBaseXC(self, other)
        else:
            raise ValueError("BaseXC can only be multiplied with float or tensor")

    def __rmul__(self, other: Union[float, int, torch.Tensor]):
        return self.__mul__(other)


def _weight_potinfo(pot: ValGrad, grid) -> ValGrad:
    # convert a plain potential to a weight-included one (see ValGrad.weighted)
    assert grid is not None, \
        "Cannot combine weighted and plain potentials without a grid"
    dvol = grid.get_dvolume().to(pot.value.dtype)

    def mulw(t):
        return None if t is None else t * dvol

    return ValGrad(
        value=pot.value * dvol,
        grad=mulw(pot.grad),
        lapl=mulw(pot.lapl),
        kin=mulw(pot.kin),
        grid=pot.grid,
        weighted=True,
    )


def _add_potinfo(a: ValGrad, b: ValGrad, grid) -> ValGrad:
    # add two potentials, harmonizing the weighted flag if they differ
    if a.weighted and not b.weighted:
        b = _weight_potinfo(b, grid)
    elif b.weighted and not a.weighted:
        a = _weight_potinfo(a, grid)
    return a + b


class AddBaseXC(BaseXC):
    def __init__(self, a: BaseXC, b: BaseXC) -> None:
        self.a = a
        self.b = b
        self._family = max(a.family, b.family)

    @property
    def family(self):
        return self._family

    @overload
    def get_vxc(self, densinfo: ValGrad) -> ValGrad: ...

    @overload
    def get_vxc(self, densinfo: SpinParam[ValGrad]) -> SpinParam[ValGrad]: ...

    def get_vxc(self, densinfo):
        avxc = self.a.get_vxc(densinfo)
        bvxc = self.b.get_vxc(densinfo)

        grid = densinfo.grid if isinstance(densinfo, ValGrad) else densinfo.u.grid

        if isinstance(densinfo, ValGrad):
            return _add_potinfo(avxc, bvxc, grid)
        else:
            return SpinParam(
                u=_add_potinfo(avxc.u, bvxc.u, grid),
                d=_add_potinfo(avxc.d, bvxc.d, grid),
            )

    def get_edensityxc(
        self, densinfo: Union[ValGrad, SpinParam[ValGrad]]
    ) -> torch.Tensor:
        return self.a.get_edensityxc(densinfo) + self.b.get_edensityxc(densinfo)

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        return self.a.getparamnames(
            methodname, prefix=prefix + "a."
        ) + self.b.getparamnames(methodname, prefix=prefix + "b.")


class MulBaseXC(BaseXC):
    def __init__(self, a: BaseXC, b: Union[float, torch.Tensor]) -> None:
        self.a = a
        self.b = b
        if isinstance(b, torch.Tensor):
            msg = "XC multiplication with tensor can only be done with 1-element tensor"
            assert b.numel() == 1, msg

    @property
    def family(self):
        return self.a.family

    @overload
    def get_vxc(self, densinfo: ValGrad) -> ValGrad: ...

    @overload
    def get_vxc(self, densinfo: SpinParam[ValGrad]) -> SpinParam[ValGrad]: ...

    def get_vxc(self, densinfo):
        avxc = self.a.get_vxc(densinfo)

        if isinstance(densinfo, ValGrad):
            return avxc * self.b
        else:
            return SpinParam(u=avxc.u * self.b, d=avxc.d * self.b)

    def get_edensityxc(
        self, densinfo: Union[ValGrad, SpinParam[ValGrad]]
    ) -> torch.Tensor:
        return self.a.get_edensityxc(densinfo) * self.b

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        params = self.a.getparamnames(methodname, prefix=prefix + "a.")
        if isinstance(self.b, torch.Tensor):
            params = params + [prefix + "b"]
        return params


class ZeroXC(BaseXC):
    family = 0

    def get_edensityxc(self, densinfo: ValGrad | SpinParam[ValGrad]) -> torch.Tensor:
        if isinstance(densinfo, SpinParam):
            val_grad = densinfo.u
        else:
            val_grad = densinfo

        shape = val_grad.value.shape
        return torch.zeros(shape)

    def get_vxc(self, densinfo: ValGrad | SpinParam[ValGrad]) -> torch.Tensor:
        edensityxc = self.get_edensityxc(densinfo)  # all zeros
        if isinstance(densinfo, SpinParam):
            # all zeros
            return SpinParam(u=ValGrad(value=edensityxc), d=ValGrad(value=edensityxc))
        else:
            # all zeros
            return ValGrad(value=edensityxc)

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "get_edensityxc":
            return []
        else:
            return super().getparamnames(methodname, prefix=prefix)
