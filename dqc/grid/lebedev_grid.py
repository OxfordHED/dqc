import os
from typing import List, Sequence, Dict
import torch
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from scipy.spatial.distance import cdist

from dqc.grid.base_grid import BaseGrid
from dqc.grid.radial_grid import RadialGrid

__all__ = ["LebedevGrid", "TruncatedLebedevGrid"]


class LebedevLoader(object):
    # load the lebedev points and save the cache to save time
    caches: Dict[int, np.ndarray] = {}
    distance_caches: Dict[int, np.ndarray] = {}

    @classmethod
    def load(cls, prec: int) -> np.ndarray:
        if prec not in cls.caches:
            # load the lebedev grid points
            dset_path = os.path.join(
                os.path.split(__file__)[0],
                "..",
                "datasets",
                "lebedevquad",
                "lebedev_%03d.txt" % prec,
            )
            assert os.path.exists(dset_path), (
                "The dataset lebedev_%03d.txt does not exist" % prec
            )
            lebedev_dsets = np.loadtxt(dset_path)
            lebedev_dsets[:, :2] *= np.pi / 180  # convert the angles to radians
            # save to the cache
            cls.caches[prec] = lebedev_dsets

        return cls.caches[prec]

    @classmethod
    def distances(cls, prec: int) -> np.ndarray:
        # Todo: cache these in files more long-term
        if prec not in cls.distance_caches:
            dist_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "datasets",
                "lebedevdists",
                "lebedev_%03d.txt" % prec,
            )
            if os.path.exists(dist_path):
                distances = np.loadtxt(dist_path)
            else:
                lebedev_dsets = cls.load(prec)
                lat_long = torch.tensor(lebedev_dsets[:, :2], dtype=torch.float64)
                lat_long[:, 1] -= torch.pi / 2  # different convention for coordinates
                lat_long = lat_long.flip(1)  # flip the coordinates
                # calculate the distances
                distances = torch.tensor(
                    haversine_distances(lat_long), dtype=torch.float64
                ) + 100 * torch.eye(lat_long.shape[0], dtype=torch.float64)
                np.savetxt(dist_path, distances.numpy())
            cls.distance_caches[prec] = torch.tensor(distances).clone().detach()

        return cls.distance_caches[prec]


class LebedevGrid(BaseGrid):
    """
    Using Lebedev predefined angular points + radial grid to form 3D grid.
    """

    def __init__(self, radgrid: RadialGrid, prec: int) -> None:
        self._dtype = radgrid.dtype
        self._device = radgrid.device
        self._prec = prec
        self._nrad = radgrid.get_rgrid().shape[0]
        self._graph = None

        assert (prec % 2 == 1) and (
            3 <= prec <= 131
        ), "Precision must be an odd number between 3 and 131"

        # load the Lebedev grid points
        lebedev_dsets = torch.tensor(
            LebedevLoader.load(prec), dtype=self._dtype, device=self._device
        )
        wphitheta = lebedev_dsets[:, -1]  # (nphitheta)
        phi = lebedev_dsets[:, 0]
        theta = lebedev_dsets[:, 1]

        # get the radial grid
        assert radgrid.coord_type == "radial"
        r = radgrid.get_rgrid().unsqueeze(-1)  # (nr, 1)

        # get the cartesian coordinate
        rsintheta = r * torch.sin(theta)
        x = (rsintheta * torch.cos(phi)).view(-1, 1)  # (nr * nphitheta, 1)
        y = (rsintheta * torch.sin(phi)).view(-1, 1)
        z = (r * torch.cos(theta)).view(-1, 1)
        xyz = torch.cat((x, y, z), dim=-1)  # (nr * nphitheta, ndim)
        self._xyz = xyz

        # calculate the dvolume (integration weights)
        dvol_rad = radgrid.get_dvolume().unsqueeze(-1)  # (nr, 1)
        self._dvolume = (dvol_rad * wphitheta).view(-1)  # (nr * nphitheta)

    def get_rgrid(self) -> torch.Tensor:
        return self._xyz

    def get_dvolume(self) -> torch.Tensor:
        return self._dvolume

    @property
    def coord_type(self) -> str:
        return "cart"

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "get_rgrid":
            return [prefix + "_xyz"]
        elif methodname == "get_dvolume":
            return [prefix + "_dvolume"]
        else:
            raise KeyError("Invalid methodname: %s" % methodname)

    @property
    def graph(self):
        if self._graph is None:
            raise Exception("Graph not generated yet")
        return self._graph

    def generate_graph(
        self,
        graph_method: str = "grid_neighbours",
        sparse: bool = True,
        range_modifier: float = 0.85,
    ) -> torch.Tensor:
        # Todo: decide whether we should include self adjacency

        # Range modifier derived from average degree scan and different precisions
        if graph_method != "grid_neighbours":
            raise ValueError("Invalid graph method: %s" % graph_method)
        # Todo: treat non-sparse case
        if not sparse:
            raise ValueError(
                "Only sparse Lebedev graph is supported due to computational cost"
            )
        distances = LebedevLoader.distances(self._prec)
        min_dists, min_inds = distances.min(dim=1, keepdim=True)
        constituent_graphs = []

        shell_graph = torch.argwhere(
            torch.triu(distances - min_dists < range_modifier * min_dists)
        )

        # Generate radial connections
        nangs = distances.shape[0]
        nrad = self._nrad
        radial_list = torch.arange(nangs, dtype=torch.int64, device=self._device)
        radial_edge_proto = radial_list.unsqueeze(1).repeat(1, 2)
        radial_edge_proto[:, 1] += nangs

        for i in range(nrad):
            constituent_graphs.append(shell_graph + i * nangs)
            if i < nrad - 1:
                constituent_graphs.append(radial_edge_proto + i * nangs)

        self._graph = torch.cat(constituent_graphs, dim=0)
        return self._graph


class TruncatedLebedevGrid(LebedevGrid):
    # A class to represent the truncated lebedev grid
    # It is represented by various radial grid (usually the sliced ones)
    # with different precisions
    def __init__(self, radgrids: Sequence[RadialGrid], precs: Sequence[int]):
        assert len(radgrids) == len(precs)
        assert len(precs) > 0
        self.lebedevs = [
            LebedevGrid(radgrid, prec) for (radgrid, prec) in zip(radgrids, precs)
        ]
        grid0 = self.lebedevs[0]

        # set the variables to be used in the properties
        self._dtype = grid0.dtype
        self._device = grid0.device
        self._xyz = torch.cat(tuple(grid.get_rgrid() for grid in self.lebedevs), dim=0)
        self._dvolume = torch.cat(
            tuple(grid.get_dvolume() for grid in self.lebedevs), dim=0
        )

    def generate_graph(
        self,
        graph_method: str = "grid_neighbours",
        sparse: bool = True,
        range_modifier: float = 0.85,
    ) -> torch.Tensor:
        outer_last = None
        graph_edges = []

        running_counter = 0
        for i, grid in enumerate(self.lebedevs):
            slice_graph = grid.generate_graph(graph_method, sparse, range_modifier)
            points_per_shell = LebedevLoader.load(grid._prec).shape[0]
            inner = grid.get_rgrid()[:points_per_shell]
            if i > 0:
                dists = cdist(inner.numpy(), outer_last.numpy())
                # Todo: currently this only picks neighbours in one direction. Do we want both?
                min_indices = np.argmin(dists, axis=1)
                pairs = torch.stack(
                    [
                        torch.tensor((min_ind, j))
                        for j, min_ind in enumerate(min_indices)
                    ],
                    dim=0,
                )

                pairs[:, 0] += running_counter
                running_counter += outer_last.shape[0]
                pairs[:, 1] += running_counter

                graph_edges.append(pairs)

            graph_edges.append(slice_graph + running_counter)
            outer_last = grid.get_rgrid()[-points_per_shell:]
            running_counter += grid.get_rgrid().shape[0] - points_per_shell
        self._graph = torch.cat(graph_edges, dim=0)
        return self._graph
