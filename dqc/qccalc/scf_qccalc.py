from __future__ import annotations
from abc import abstractmethod, abstractproperty
from typing import Optional, Dict, Any, List, Union, Tuple
import re
import warnings
import torch
import xitorch as xt
import xitorch.linalg
import xitorch.optimize
from dqc.system.base_system import BaseSystem
from dqc.qccalc.base_qccalc import BaseQCCalc
from dqc.utils.datastruct import SpinParam
from dqc.utils.config import config
from dqc.utils.misc import set_default_option

class SCF_QCCalc(BaseQCCalc):
    """
    Performing Restricted or Unrestricted self-consistent field iteration
    (e.g. Hartree-Fock or Density Functional Theory)

    Arguments
    ---------
    engine: BaseSCFEngine
        The SCF engine
    variational: bool
        If True, then use optimization of the free orbital parameters to find
        the minimum energy.
        Otherwise, use self-consistent iterations.
    """

    def __init__(self, engine: BaseSCFEngine, variational: bool = False):
        self.density_hist = None
        self._engine = engine
        self._polarized = engine.polarized
        self._shape = self._engine.shape
        self.dtype = self._engine.dtype
        self.device = self._engine.device
        self._has_run = False
        self._variational = variational
        # SCF convergence diagnostics, populated by run() (None until then)
        self._scf_diag: Optional[Dict[str, Any]] = None
        self._solve_warnings: List[str] = []

    def get_system(self) -> BaseSystem:
        return self._engine.get_system()

    def run(self, dm0: Optional[Union[str, torch.Tensor, SpinParam[torch.Tensor]]] = "1e",  # type: ignore
            eigen_options: Optional[Dict[str, Any]] = None,
            fwd_options: Optional[Dict[str, Any]] = None,
            bck_options: Optional[Dict[str, Any]] = None,
            return_history: bool = False,
            diagnose: bool = False,
            conv_tol: float = 1e-6,
            strict: bool = False) -> BaseQCCalc:
        """Run the self-consistent field iteration and return the converged calculation.

        Arguments
        ---------
        dm0: str or torch.Tensor or SpinParam of torch.Tensor
            Initial density-matrix guess. ``"1e"`` (default) builds it from the
            one-electron Hamiltonian.
        eigen_options: dict or None
            Options forwarded to the orbital diagonalization (default
            ``{"method": "exacteig"}``).
        fwd_options: dict or None
            Options for the forward SCF fixed-point solver.
        bck_options: dict or None
            Options for the implicit (backward) solve used by autograd.
        return_history: bool
            If True, keep the density-matrix history on ``self.density_hist``.
            This is for diagnostics only -- it breaks ``backward()`` through the
            solve, so do not combine it with gradient-carrying runs.
        diagnose: bool
            If True, after the solve recompute the relative SCF fixed-point
            residual ||scp2scp(scp) - scp|| / ||scp|| and store a convergence
            diagnostic (see ``get_scf_diagnostics``). Costs ~1 extra SCF step.
        conv_tol: float
            Relative residual below which the SCF is deemed converged. Default
            1e-6: the energy error scales as residual^2, so 1e-6 is
            energy-converged while still flagging genuine non-convergence
            (stalls/divergence give residuals of 1e-3..1e0). Tighten for
            gradient-sensitive work; loosen to silence marginal cases.
        strict: bool
            If True, raise RuntimeError when the SCF did not converge (else warn).
        """
        self._solve_warnings = []
        # get default options
        if not self._variational:
            fwd_defopt = {
                "method": "broyden1",
                "alpha": -0.5,
                "maxiter": 50,
                "verbose": config.VERBOSE > 0,
            }
        else:
            fwd_defopt = {
                "method": "gd",
                "step": 1e-2,
                "maxiter": 5000,
                "f_rtol": 1e-10,
                "x_rtol": 1e-10,
                "verbose": config.VERBOSE > 0,
            }
        bck_defopt = {
            # NOTE: it seems like in most cases the jacobian matrix is posdef
            # if it is not the case, we can just remove the line below
            "posdef": True,
        }

        # setup the default options
        if eigen_options is None:
            eigen_options = {
                "method": "exacteig"
            }
        if fwd_options is None:
            fwd_options = {}
        if bck_options is None:
            bck_options = {}
        fwd_options = set_default_option(fwd_defopt, fwd_options)
        bck_options = set_default_option(bck_defopt, bck_options)

        # save the eigen_options for use in diagonalization
        self._engine.set_eigen_options(eigen_options)

        # set up the initial self-consistent param guess
        if dm0 is None:
            dm = self._get_zero_dm()
        elif isinstance(dm0, str):
            if dm0 == "1e":  # initial density based on 1-electron Hamiltonian
                dm = self._get_zero_dm()
                scp0 = self._engine.dm2scp(dm)
                dm = self._engine.scp2dm(scp0)
            else:
                raise RuntimeError("Unknown dm0: %s" % dm0)
        else:
            dm = SpinParam.apply_fcn(lambda dm0: dm0.detach(), dm0)

        # making it spin param for polarized and tensor for nonpolarized
        if isinstance(dm, torch.Tensor) and self._polarized:
            dm_u = dm * 0.5
            dm_d = dm * 0.5
            dm = SpinParam(u=dm_u, d=dm_d)
        elif isinstance(dm, SpinParam) and not self._polarized:
            dm = dm.u + dm.d

        if not self._variational:
            scp0 = self._engine.dm2scp(dm)

            # Grad-carrying solve. NOTE: we deliberately do NOT pass
            # return_history here -- with return_history=True xitorch.equilibrium
            # returns a (scp, history) tuple, which is incompatible with
            # backward() (the autograd Function then has multiple outputs and
            # _RootFinder.backward accepts only one grad). Capture xitorch's
            # ConvergenceWarning (otherwise silently discarded).
            with warnings.catch_warnings(record=True) as _wlist:
                warnings.simplefilter("always")
                scp = xitorch.optimize.equilibrium(
                    fcn=self._engine.scp2scp,
                    y0=scp0,
                    bck_options={**bck_options},
                    **fwd_options)
            self._solve_warnings = [str(w.message) for w in _wlist]

            # The SCF trajectory is inspection-only (non-differentiable), so when
            # requested we recover it with a detached re-solve of the same
            # deterministic iteration -- this keeps return_history compatible
            # with autograd, unlike passing it into the grad-carrying solve.
            if return_history:
                with torch.no_grad():
                    sc_hist_result = xitorch.optimize.equilibrium(
                        fcn=self._engine.scp2scp,
                        y0=scp0.detach(),
                        bck_options={**bck_options},
                        return_history=True,
                        **fwd_options)
                if isinstance(sc_hist_result, tuple):
                    _, scp_history = sc_hist_result
                    self.density_hist = [self._engine.scp2dm(x) for x in scp_history]
                # else: xitorch 0.5.x ignores return_history; density_hist stays None

            # post-process parameters
            self._dm = self._engine.scp2dm(scp)

        else:
            system = self.get_system()
            h = system.get_hamiltonian()
            orb_weights = system.get_orbweight(polarized=self._polarized)
            norb = SpinParam.apply_fcn(lambda orb_weights: len(orb_weights), orb_weights)

            def dm2params(dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> \
                    Tuple[torch.Tensor, torch.Tensor]:
                pc = SpinParam.apply_fcn(
                     lambda dm, norb: h.dm2ao_orb_params(SpinParam.sum(dm), norb=norb),
                     dm, norb)
                p = SpinParam.apply_fcn(lambda pc: pc[0], pc)
                c = SpinParam.apply_fcn(lambda pc: pc[1], pc)
                params = self._engine.pack_aoparams(p)
                coeffs = self._engine.pack_aoparams(c)
                return params, coeffs

            def params2dm(params: torch.Tensor, coeffs: torch.Tensor) \
                    -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
                p: Union[torch.Tensor, SpinParam[torch.Tensor]] = self._engine.unpack_aoparams(params)
                c: Union[torch.Tensor, SpinParam[torch.Tensor]] = self._engine.unpack_aoparams(coeffs)

                dm = SpinParam.apply_fcn(
                    lambda p, c, orb_weights: h.ao_orb_params2dm(p, c, orb_weights, with_penalty=None),
                    p, c, orb_weights)
                return dm

            params0, coeffs0 = dm2params(dm)
            params0 = params0.detach()
            coeffs0 = coeffs0.detach()
            min_params0: torch.Tensor = xitorch.optimize.minimize(
                fcn=self._engine.aoparams2ene,
                # random noise to add the chance of it gets to the minimum, not
                # a saddle point
                y0=params0 + torch.randn_like(params0) * 0.03 / params0.numel(),
                params=(coeffs0, None,),  # coeffs & with_penalty
                bck_options={**bck_options},
                **fwd_options).detach()

            if torch.is_grad_enabled():
                # If the gradient is required, then put it through the minimization
                # one more time with penalty on the parameters.
                # The penalty is to keep the Hamiltonian invertible, stabilizing
                # inverse.
                # Without the penalty, the Hamiltonian could have 0 eigenvalues
                # because of the overparameterization of the aoparams.
                min_dm = params2dm(min_params0, coeffs0)
                params0, coeffs0 = dm2params(min_dm)
                min_params0 = xitorch.optimize.minimize(
                    fcn=self._engine.aoparams2ene,
                    y0=params0,
                    params=(coeffs0, 1e-1,),  # coeffs & with_penalty
                    bck_options={**bck_options},
                    method="gd",
                    step=0,
                    maxiter=0)

            self._dm = params2dm(min_params0, coeffs0)

        self._has_run = True

        # SCF convergence diagnostics (the signal DQC previously discarded).
        # Guarded so a diagnostic failure can never break an otherwise-good run.
        if diagnose:
            try:
                self._scf_diag = self._compute_scf_diag(conv_tol)
            except Exception as exc:  # pragma: no cover - defensive
                self._scf_diag = {"converged": None, "residual": float("nan"),
                                  "niter": None, "conv_tol": conv_tol,
                                  "warnings": [f"scf diagnostic failed: {exc}"]}
            if self._scf_diag["converged"] is False:
                niter = self._scf_diag["niter"]
                msg = ("SCF did not converge: relative fixed-point residual "
                       f"{self._scf_diag['residual']:.3e} > conv_tol {conv_tol:.1e}"
                       + (f" ({niter} iterations)" if niter is not None else ""))
                if strict:
                    raise RuntimeError(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)

        return self

    def _compute_scf_diag(self, conv_tol: float) -> Dict[str, Any]:
        """Recompute the relative SCF fixed-point residual at the final density
        and assemble a convergence diagnostic. Detached: no autograd cost."""
        dm = self._dm
        if isinstance(dm, SpinParam):
            dm_d: Union[torch.Tensor, SpinParam] = SpinParam(u=dm.u.detach(), d=dm.d.detach())
        else:
            dm_d = dm.detach()
        with torch.no_grad():
            scp = self._engine.dm2scp(dm_d)
            scp_next = self._engine.scp2scp(scp)
            residual = float((scp_next - scp).norm() / (scp.norm() + 1e-30))

        conv_warn = [w for w in self._solve_warnings if "converg" in w.lower()]
        niter: Optional[int] = None
        if self.density_hist is not None:
            niter = len(self.density_hist)
        elif conv_warn:
            m = re.search(r"after (\d+) iterations", conv_warn[0])
            if m:
                niter = int(m.group(1))
        return {
            "converged": residual < conv_tol,
            "residual": residual,
            "niter": niter,
            "conv_tol": conv_tol,
            "warnings": conv_warn,
        }

    def is_converged(self) -> bool:
        """Whether the SCF reached the requested fixed-point tolerance.

        Only meaningful if run(diagnose=True). Returns False if diagnostics
        were disabled.
        """
        assert self._has_run, "run() must be called first"
        return bool(self._scf_diag is not None and self._scf_diag["converged"] is True)

    def get_scf_diagnostics(self) -> Optional[Dict[str, Any]]:
        """Return the SCF convergence diagnostic dict (or None if diagnose=False):

            {converged: bool, residual: float (relative fixed-point residual),
             niter: int|None, conv_tol: float, warnings: list[str]}
        """
        assert self._has_run, "run() must be called first"
        return self._scf_diag

    def energy(self) -> torch.Tensor:
        """Total electronic energy E."""
        assert self._has_run
        return self._engine.dm2energy(self._dm)

    def aodm(self) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        # returns the density matrix in the atomic-orbital basis
        assert self._has_run
        return self._dm

    def total_aodm(self) -> torch.Tensor:
        assert self._has_run
        aodm = self.aodm()
        return aodm if not isinstance(aodm, SpinParam) else (aodm.u + aodm.d)

    def get_rho(self) -> torch.Tensor:
        assert self._has_run
        aodmt = self.total_aodm()
        ham = self.get_system().get_hamiltonian()
        return ham.aodm2dens(aodmt, ham.grid.get_rgrid())

    def dm2energy(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]):
        # calculate the energy given the density matrix
        assert (isinstance(dm, torch.Tensor) and not self._polarized) or \
            (isinstance(dm, SpinParam) and self._polarized)
        return self._engine.dm2energy(dm)

    def _get_zero_dm(self) -> Union[SpinParam[torch.Tensor], torch.Tensor]:
        # get the initial dm that are all zeros
        if not self._polarized:
            return torch.zeros(self._shape, dtype=self.dtype,
                               device=self.device)
        else:
            dm0_u = torch.zeros(self._shape, dtype=self.dtype,
                                device=self.device)
            dm0_d = torch.zeros(self._shape, dtype=self.dtype,
                                device=self.device)
            return SpinParam(u=dm0_u, d=dm0_d)

    def get_density_hist(self) -> None | list[torch.Tensor]:
        # get list of densities from each iteration of the self-consistency cycle in form of dm
        assert self._has_run
        return self.density_hist

class BaseSCFEngine(xt.EditableModule):
    @abstractproperty
    def polarized(self) -> bool:
        """
        Returns if the system is polarized or not
        """
        pass

    @abstractproperty
    def shape(self):
        """
        Returns the shape of the density matrix in this engine.
        """
        pass

    @abstractproperty
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the tensors in this engine.
        """
        pass

    @abstractproperty
    def device(self) -> torch.device:
        """
        Returns the device of the tensors in this engine.
        """
        pass

    @abstractmethod
    def get_system(self) -> BaseSystem:
        """
        Returns the system involved in the engine.
        """
        pass

    @abstractmethod
    def dm2energy(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        """
        Calculate the energy from the given density matrix.
        """
        pass

    @abstractmethod
    def dm2scp(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        """
        Convert the density matrix into the self-consistent parameter (scp).
        Self-consistent parameter is defined as the parameter that is put into
        the equilibrium function, i.e. y in `y = f(y, x)`.
        """
        pass

    @abstractmethod
    def scp2dm(self, scp: torch.Tensor) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        """
        Calculate the density matrix from the given self-consistent parameter (scp).
        """
        pass

    @abstractmethod
    def scp2scp(self, scp: torch.Tensor) -> torch.Tensor:
        """
        Calculate the next self-consistent parameter (scp) for the next iteration
        from the previous scp.
        """
        pass

    @abstractmethod
    def aoparams2ene(self, aoparams: torch.Tensor, aocoeffs: torch.Tensor,
                     with_penalty: Optional[float] = None) -> torch.Tensor:
        """
        Calculate the energy from the given atomic orbital parameters and coefficients.
        """
        pass

    @abstractmethod
    def aoparams2dm(self, aoparams: torch.Tensor, aocoeffs: torch.Tensor,
                    with_penalty: Optional[float] = None) -> \
            Tuple[Union[torch.Tensor, SpinParam[torch.Tensor]], Optional[torch.Tensor]]:
        """
        Calculate the density matrix and the penalty from the given atomic
        orbital parameters and coefficients.
        """
        pass

    @abstractmethod
    def pack_aoparams(self, aoparams: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        """
        Pack the ao params into a single tensor.
        """
        pass

    @abstractmethod
    def unpack_aoparams(self, aoparams: torch.Tensor) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        """
        Unpack the ao params into a tensor or SpinParam of tensor.
        """
        pass

    @abstractmethod
    def set_eigen_options(self, eigen_options: Dict[str, Any]) -> None:
        """
        Set the options for the diagonalization (i.e. eigendecomposition).
        """
        pass

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """
        List all the names of parameters used in the given method.
        """
        pass
