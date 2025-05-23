from __future__ import annotations
from typing import List, Union, Optional, Tuple, Dict, Any
from math import ldexp
import warnings
import torch
from dqc.hamilton.base_hamilton import BaseHamilton
from dqc.hamilton.hcgto import HamiltonCGTO
from dqc.system.base_system import BaseSystem
from dqc.grid.base_grid import BaseGrid
from dqc.grid.factory import get_predefined_grid
from dqc.utils.datastruct import (
    CGTOBasis,
    AtomCGTOBasis,
    ValGrad,
    SpinParam,
    ZType,
    is_z_float,
    BasisInpType,
    DensityFitInfo,
    AtomZsType,
    AtomPosType,
)
from dqc.utils.periodictable import get_atomz, get_atom_mass
from dqc.utils.safeops import occnumber, safe_cdist
from dqc.api.loadbasis import loadbasis
from dqc.api.parser import parse_moldesc
from dqc.utils.cache import Cache
from dqc.utils.misc import logger

__all__ = ["Mol"]


class MolEmbedding:
    def __init__(self, mol: "Mol"):
        # Todo: How do we get distinction between same atom grid or different atom grid?
        # (without violating permutation invariance)
        try:
            grid = mol.get_grid()
        except RuntimeError:
            mol.setup_grid()
            grid = mol.get_grid()

        radial_dists = [
            torch.norm(atomg.get_rgrid(), p=2, dim=1) for atomg in grid._atomgrids
        ]
        self._radial_dists = torch.cat(radial_dists, dim=0)
        len_subgrids = torch.tensor([rg.shape[0] for rg in radial_dists])
        self._atom_zs = torch.repeat_interleave(mol.atomzs, len_subgrids)
        self._chunk_tracker = 0
        del grid._atomgrids

    def apply(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        if isinstance(densinfo, SpinParam):
            dens = SpinParam.sum(densinfo).value
            zeta = (densinfo.u.value - densinfo.d.value) / torch.where(
                dens > ldexp(1.0, -53), dens, ldexp(1.0, -53)
            )
        else:
            dens = densinfo.value
            zeta = torch.zeros_like(dens)

        return torch.stack([dens, zeta, self._radial_dists, self._atom_zs], dim=-1)


class Mol(BaseSystem):
    """
    Describe the system of an isolated molecule.

    Arguments
    ---------
    * moldesc: str or 2-elements tuple
        Description of the molecule system.
        If string, it can be described like ``"H 1 0 0; H -1 0 0"``.
        If tuple, the first element of the tuple is the Z number of the atoms while
        the second element is the position of the atoms: ``(atomzs, atomposs)``.
    * basis: str, CGTOBasis, list of str, or CGTOBasis
        The string describing the gto basis. If it is a list, then it must have
        the same length as the number of atoms.
    * grid: int
        Describe the grid.
        If it is an integer, then it uses the default grid with specified level
        of accuracy.
    * spin: int, float, torch.Tensor, or None
        The difference between spin-up and spin-down electrons.
        It must be an integer or ``None``.
        If ``None``, then it is ``num_electrons % 2``.
        For floating point atomzs and/or charge, the ``spin`` must be specified.
    * charge: int, float, or torch.Tensor
        The charge of the molecule.
    * orb_weights: SpinParam[torch.Tensor] or None
        Specifiying the orbital occupancy (or weights) directly. If specified,
        ``spin`` and ``charge`` arguments are ignored.
    * vext: tensor or None
        The tensor describing the external potential given in the grid.
        The grid position can be obtained by ``Mol().get_grid().get_rgrid()``.
    * efield: tensor, tuple of tensor, or None
        Uniform electric field of the system. If a tensor, then it is assumed
        to be a constant electric field with the energy is
        calculated based on potential at ``(0, 0, 0)`` is ``0``.
        If a tuple of tensor, then the first element will have a shape of ``(ndim,)``
        representing the constant electric field, second element is the gradient
        of electric field with the last dimension is the direction of the electric
        field, third element is the gradgrad of electric field, etc.
        If ``None``, then the electric field is assumed to be ``0``.
    * dtype: torch.dtype
        The data type of tensors in this class.
    * device: torch.device
        The device on which the tensors in this class are stored.

    * orthogonalize_basis: bool
        (computational option)
        If True, orthogonalize the basis in the hamiltonian calculation.
        If False, then use the raw basis, this might not work with over-complete
        basis.
    * ao_parameterizer: str
        (computational option)
        Specifying the atomic orbital parameterizer.
    """

    def __init__(
        self,
        moldesc: Union[str, Tuple[AtomZsType, AtomPosType]],
        basis: BasisInpType,
        *,
        orthogonalize_basis: bool = True,
        ao_parameterizer: str = "qr",
        grid: Union[int, str] = "sg3",
        spin: Optional[ZType] = None,
        charge: ZType = 0,
        orb_weights: Optional[SpinParam[torch.Tensor]] = None,
        efield: Union[torch.Tensor, Tuple[torch.Tensor, ...], None] = None,
        vext: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
        graph: str | torch.Tensor | None = None,
        graph_kwargs: dict[str, Any] | None = None,
        grid_params: dict | None = None,
    ):
        self._grid_params = grid_params
        self._dtype = dtype
        self._device = device
        self._grid_inp = grid
        self._basis_inp = basis
        self._grid: Optional[BaseGrid] = None
        self._vext = vext
        self._graph = graph
        self._graph_kwargs = graph_kwargs or {}
        self._embedding = None

        # make efield a tuple
        self._efield = _normalize_efield(efield)
        self._preproc_efield = _preprocess_efield(self._efield)

        # initialize cache
        self._cache = Cache()

        # get the AtomCGTOBasis & the hamiltonian
        # atomzs: (natoms,) dtype: torch.int or dtype for floating point
        # atompos: (natoms, ndim)
        atomzs, atompos = parse_moldesc(moldesc, dtype=dtype, device=device)
        atomzs_int = (
            torch.round(atomzs).to(torch.int) if atomzs.is_floating_point() else atomzs
        )
        allbases = _parse_basis(atomzs_int, basis)  # list of list of CGTOBasis
        atombases = [
            AtomCGTOBasis(atomz=atz, bases=bas, pos=atpos)
            for (atz, bas, atpos) in zip(atomzs, allbases, atompos)
        ]
        self._atombases = atombases
        self._hamilton = HamiltonCGTO(
            atombases,
            efield=self._preproc_efield,
            vext=self._vext,
            cache=self._cache.add_prefix("hamilton"),
            orthozer=orthogonalize_basis,
            aoparamzer=ao_parameterizer,
        )
        self._orthogonalize_basis = orthogonalize_basis
        self._aoparamzer = ao_parameterizer
        self._atompos = atompos  # (natoms, ndim)
        self._atomzs = atomzs  # (natoms,) int-type or dtype if floating point
        self._atomzs_int = atomzs_int  # (natoms,) int-type rounded from atomzs
        nelecs_tot: torch.Tensor = torch.sum(atomzs)

        # orb_weights is not specified, so determine it from spin and charge
        if orb_weights is None:
            # get the number of electrons and spin
            nelecs, spin, frac_mode = _get_nelecs_spin(nelecs_tot, spin, charge)
            _orb_weights, _orb_weights_u, _orb_weights_d = _get_orb_weights(
                nelecs, spin, frac_mode, dtype, device
            )

            # save the system's properties
            self._spin = spin
            self._charge = charge
            self._numel = nelecs
            self._orb_weights = _orb_weights
            self._orb_weights_u = _orb_weights_u
            self._orb_weights_d = _orb_weights_d

        # orb_weights is specified, so calculate the spin and charge from it
        else:
            if not isinstance(orb_weights, SpinParam):
                raise TypeError("Specifying orb_weights must be in SpinParam type")
            assert orb_weights.u.ndim == 1
            assert orb_weights.d.ndim == 1
            assert len(orb_weights.u) == len(orb_weights.d)

            # check if it is decreasing
            orb_u_dec = torch.all(orb_weights.u[:-1] - orb_weights.u[1:] > -1e-4)
            orb_d_dec = torch.all(orb_weights.d[:-1] - orb_weights.d[1:] > -1e-4)
            if not (orb_u_dec and orb_d_dec):
                # if not decreasing, the variational might give the wrong results
                warnings.warn(
                    "The orbitals should be ordered in a non-increasing manner. "
                    "Otherwise, some calculations might be wrong."
                )

            utot = orb_weights.u.sum()
            dtot = orb_weights.d.sum()
            self._numel = utot + dtot
            self._spin = utot - dtot
            self._charge = nelecs_tot - self._numel

            self._orb_weights_u = orb_weights.u
            self._orb_weights_d = orb_weights.d
            self._orb_weights = orb_weights.u + orb_weights.d

    def densityfit(
        self, method: Optional[str] = None, auxbasis: Optional[BasisInpType] = None
    ) -> BaseSystem:
        """
        Indicate that the system's Hamiltonian uses density fit for its integral.

        Arguments
        ---------
        method: Optional[str]
            Density fitting method. Available methods in this class are:

            * ``"coulomb"``: Minimizing the Coulomb inner product, i.e. ``min <p-p_fit|r_12|p-p_fit>``
              Ref: Eichkorn, et al. Chem. Phys. Lett. 240 (1995) 283-290.
              (default)
            * ``"overlap"``: Minimizing the overlap inner product, i.e. min <p-p_fit|p-p_fit>

        auxbasis: Optional[BasisInpType]
            Auxiliary basis for the density fit. If not specified, then it uses
            ``"cc-pvtz-jkfit"``.
        """
        if method is None:
            method = "coulomb"
        if auxbasis is None:
            # TODO: choose the auxbasis properly
            auxbasis = "cc-pvtz-jkfit"

        # get the auxiliary basis
        assert auxbasis is not None
        auxbasis_lst = _parse_basis(self._atomzs_int, auxbasis)
        atomauxbases = [
            AtomCGTOBasis(atomz=atz, bases=bas, pos=atpos)
            for (atz, bas, atpos) in zip(self._atomzs, auxbasis_lst, self._atompos)
        ]

        # change the hamiltonian to have density fit
        df = DensityFitInfo(method=method, auxbases=atomauxbases)
        self._hamilton = HamiltonCGTO(
            self._atombases,
            df=df,
            efield=self._preproc_efield,
            vext=self._vext,
            cache=self._cache.add_prefix("hamilton"),
            orthozer=self._orthogonalize_basis,
            aoparamzer=self._aoparamzer,
        )
        return self

    def get_hamiltonian(self) -> BaseHamilton:
        """
        Returns the Hamiltonian that corresponds to the system, i.e.
        :class:`~dqc.hamilton.HamiltonCGTO`
        """
        return self._hamilton

    def set_cache(
        self, fname: str, paramnames: Optional[List[str]] = None
    ) -> BaseSystem:
        """
        Setup the cache of some parameters specified by `paramnames` to be read/written
        on a file.
        If the file exists, then the parameters will not be recomputed, but just
        loaded from the cache instead.
        Cache is usually used for repeated calculations where the cached parameters
        are not changed (e.g. running multiple systems with slightly different environment.)

        Arguments
        ---------
        fname: str
            The file to store the cache.
        paramnames: list of str or None
            List of parameter names to be read/write from the cache.
        """
        all_paramnames = self._cache.get_cacheable_params()
        if paramnames is not None:
            # check the paramnames
            for pname in paramnames:
                if pname not in all_paramnames:
                    msg = (
                        "Parameter %s is not cache-able. Cache-able parameters are %s"
                        % (pname, all_paramnames)
                    )
                    raise ValueError(msg)

        self._cache.set(fname, paramnames)

        return self

    def get_orbweight(
        self, polarized: bool = False
    ) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        if not polarized:
            return self._orb_weights
        else:
            return SpinParam(u=self._orb_weights_u, d=self._orb_weights_d)

    def get_nuclei_energy(self) -> torch.Tensor:
        # atomzs: (natoms,)
        # atompos: (natoms, ndim)

        # r12: (natoms, natom)
        r12 = safe_cdist(self._atompos, self._atompos, add_diag_eps=True, diag_inf=True)
        z12 = self._atomzs.unsqueeze(-2) * self._atomzs.unsqueeze(
            -1
        )  # (natoms, natoms)
        q_by_r = z12 / r12
        return q_by_r.sum() * 0.5

    def setup_grid(self) -> None:
        if self._grid is not None:
            return self._grid
        logger.log("Constructing the integration grid")
        kwargs = {"dtype": self._dtype, "device": self._device}
        if self._grid_params is not None:
            kwargs = {**self._grid_params, **kwargs}
        self._grid = get_predefined_grid(
            self._grid_inp, self._atomzs_int, self._atompos, **kwargs
        )
        if isinstance(self._graph, str):
            self._grid.generate_graph(self._graph, **self._graph_kwargs)
        elif isinstance(self._graph, torch.Tensor):
            self._grid._graph = self._graph
        logger.log("Constructing the integration grid: done")

    def get_grid(self) -> BaseGrid:
        if self._grid is None:
            raise RuntimeError(
                "Please run mol.setup_grid() first before calling get_grid()"
            )
        return self._grid

    def get_graph(self) -> Optional[Tuple[torch.Tensor, ...]]:
        if self._grid is None:
            raise RuntimeError(
                "Please run mol.setup_grid() first before calling get_graph()"
            )
        return self._grid.graph

    def get_embedding(self) -> MolEmbedding:
        if self._embedding is None:
            self._embedding = MolEmbedding(self)
        return self._embedding

    def requires_grid(self) -> bool:
        req_grid = self._vext is not None
        return req_grid

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "get_nuclei_energy":
            params = [prefix + "_atompos"]
            if torch.is_floating_point(self._atomzs):
                params += [prefix + "_atomzs"]
            return params
        else:
            raise KeyError("Unknown methodname: %s" % methodname)

    def make_copy(self, **kwargs) -> Mol:
        """
        Returns a copy of the system identical to the orginal except for new
        parameters set in the kwargs.

        Arguments
        ---------
        **kwargs
            Must be the same kwargs as Mol.
        """
        # create dictionary of all parameters
        parameters = {
            "moldesc": (self.atomzs, self.atompos),
            "basis": self._basis_inp,
            "orthogonalize_basis": self._orthogonalize_basis,
            "ao_parameterizer": self._aoparamzer,
            "grid": self._grid_inp,
            "spin": self._spin,
            "charge": self._charge,
            "orb_weights": None,
            "efield": self._efield,
            "vext": self._vext,
            "dtype": self._dtype,
            "device": self._device,
        }
        # update dictionary with provided kwargs
        parameters.update(kwargs)
        # create new system
        return Mol(**parameters)

    ################### properties ###################
    @property
    def atompos(self) -> torch.Tensor:
        return self._atompos

    @property
    def atomzs(self) -> torch.Tensor:
        return self._atomzs

    @property
    def atommasses(self) -> torch.Tensor:
        # returns the atomic mass (only for non-isotope for now)
        if torch.is_floating_point(self._atomzs):
            raise RuntimeError("Atom masses are not available for floating point Z")
        return torch.tensor(
            [get_atom_mass(int(atomz)) for atomz in self._atomzs],
            dtype=self._dtype,
            device=self._device,
        )

    @property
    def spin(self) -> ZType:
        return self._spin

    @property
    def charge(self) -> ZType:
        return self._charge

    @property
    def numel(self) -> ZType:
        return self._numel

    @property
    def efield(self) -> Optional[Tuple[torch.Tensor, ...]]:
        return self._efield


def _parse_basis(atomzs: torch.Tensor, basis: BasisInpType) -> List[List[CGTOBasis]]:
    # returns the list of cgto basis for every atoms
    natoms = len(atomzs)

    if isinstance(basis, str):
        return [loadbasis("%d:%s" % (atomz, basis)) for atomz in atomzs]

    elif isinstance(basis, dict):
        # convert the basis key into atomz
        basis_int: Dict[int, List[CGTOBasis]] = {}
        for k, v in basis.items():
            atz = int(get_atomz(k))
            bas = v if isinstance(v, list) else loadbasis("%d:%s" % (atz, v))
            basis_int[atz] = bas
        return [basis_int[int(atomz)] for atomz in atomzs]

    else:  # basis is a list
        assert len(atomzs) == len(basis)

        # TODO: safely remove "type: ignore" in this block
        assert len(basis) > 0

        # list of cgtobasis
        if isinstance(basis[0], CGTOBasis):
            return [basis for _ in range(natoms)]  # type: ignore

        # list of str
        elif isinstance(basis[0], str):
            return [loadbasis("%d:%s" % (atz, b)) for (atz, b) in zip(atomzs, basis)]  # type: ignore

        # list of list of cgto basis
        else:
            return basis  # type: ignore


def _get_nelecs_spin(
    nelecs_tot: torch.Tensor, spin: Optional[ZType], charge: ZType
) -> Tuple[torch.Tensor, ZType, bool]:
    # get the number of electrons and spins

    # a boolean to indicate if working in a fractional mode
    frac_mode = (
        nelecs_tot.is_floating_point()
        or is_z_float(charge)
        or (spin is not None and is_z_float(spin))
    )

    assert nelecs_tot >= charge, "Only %f electrons, but needs %f charge" % (
        nelecs_tot.item(),
        charge,
    )
    nelecs: torch.Tensor = nelecs_tot - charge

    # if spin is not given, then set it as the remainder if nelecs is an integer
    if spin is None:
        assert not frac_mode, "Fraction case requires the spin argument to be specified"
        spin = nelecs % 2
    else:
        assert spin >= 0
        if not frac_mode:
            # only check if the calculation is not in fraction mode,
            # for fractional mode, unmatched spin is acceptable
            assert (nelecs - spin) % 2 == 0, (
                "Spin %d is not suited for %d electrons" % (spin, nelecs)
            )
    return nelecs, spin, frac_mode


def _get_orb_weights(
    nelecs: torch.Tensor,
    spin: ZType,
    frac_mode: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # returns the orbital weights given the electronic information
    # (total orbital weights, spin-up orb weights, spin-down orb weights)

    # calculate the orbital weights
    nspin_dn: torch.Tensor = (
        (nelecs - spin) * 0.5
        if frac_mode
        else torch.div(nelecs - spin, 2, rounding_mode="floor")
    )
    nspin_up: torch.Tensor = nspin_dn + spin

    # total orbital weights
    _orb_weights_u = occnumber(nspin_up, dtype=dtype, device=device)
    _orb_weights_d = occnumber(
        nspin_dn, n=len(_orb_weights_u), dtype=dtype, device=device
    )
    _orb_weights = _orb_weights_u + _orb_weights_d

    # get the polarized orbital weights
    if nspin_dn > 0:
        _orb_weights_d = occnumber(nspin_dn, dtype=dtype, device=device)
    else:
        _orb_weights_d = occnumber(0, n=1, dtype=dtype, device=device)

    return _orb_weights, _orb_weights_u, _orb_weights_d


def _normalize_efield(
    efield: Union[torch.Tensor, Tuple[torch.Tensor, ...], None],
) -> Optional[Tuple[torch.Tensor, ...]]:
    # making efield a tuple or None

    if isinstance(efield, torch.Tensor):
        efs: Optional[Tuple[torch.Tensor, ...]] = (efield,)
    else:
        efs = efield

    return efs


def _preprocess_efield(
    efs: Optional[Tuple[torch.Tensor, ...]],
) -> Optional[Tuple[torch.Tensor, ...]]:
    # preprocess the efield tuple such that the energy is just a simple tensor
    # product with the integrals

    if efs is None:
        return efs

    assert isinstance(efs, tuple)
    res_list: List[torch.Tensor] = []
    for i in range(len(efs)):
        efi = efs[i]
        numel = 3 ** (i + 1)
        assert (
            efi.numel() == numel
        ), f"The {i}-th tuple element of efield must have {numel} elements"

        res_list.append(efi.reshape(-1))

    return tuple(res_list)
