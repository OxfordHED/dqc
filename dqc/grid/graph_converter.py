from __future__ import annotations
from torch_cluster import radius_graph
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

mpl.use("TkAgg")
from dqc.grid.base_grid import BaseGrid
from dqc.grid.factory import get_predefined_grid


grid = get_predefined_grid("sg2", torch.tensor([1]), torch.tensor([[0., 0., 0.]]).to(float))


def make_graph(grid: BaseGrid, r: float, nmax: int) -> torch.Tensor:
    rgrid = grid.get_rgrid()
    graph = radius_graph(rgrid, r, max_num_neighbors=nmax)
    return graph

g = make_graph(grid, 5, 10)

def graph_hist(graph: torch.Tensor, num_nodes: int, n_bins: int = 50) -> tuple[list[float], np.array[int]]:
    occurrences = graph.flatten()
    unique, counts = torch.unique(occurrences, return_counts=True)
    missing_nodes = []
    for node in range(num_nodes):
        if node not in unique:
            missing_nodes.append(node)

    missing_counts = torch.zeros(len(missing_nodes))
    unique = torch.cat((unique, torch.tensor(missing_nodes)))
    counts = torch.cat((counts, missing_counts))
    assert len(unique) == num_nodes
    assert len(counts) == num_nodes
    histogram, edges = np.histogram(counts, bins=torch.arange(-.5, torch.max(counts) + 0.5, torch.max(counts) / n_bins))
    centers = [(edges[i] + edges[i+1])/2 for i in range(len(edges) - 1)]
    return centers, histogram

def edge_hist(g: torch.Tensor, rgrid: torch.Tensor):
    starting, ending = g[0], g[1]
    starting_pos = rgrid[starting, :]
    ending_pos = rgrid[ending, :]
    vectors = ending_pos - starting_pos
    lengths = torch.sqrt(torch.sum(torch.square(vectors), dim=1))
    histogram, edges = np.histogram(lengths, bins=1_000)
    centers = [(edges[i] + edges[i+1])/2 for i in range(len(edges) - 1)]
    return centers, histogram


hist = graph_hist(g, grid.get_rgrid().shape[0], n_bins=20)

fig, axs = plt.subplots(1, 2)
plt.sca(axs[0])
plt.bar(*hist, width=(hist[0][1] - hist[0][0])/2)
plt.xlabel("Number of Neighbours")
plt.ylabel("Node Count")
plt.sca(axs[1])
e_hist = edge_hist(g, grid.get_rgrid())
plt.bar(*e_hist, width=(e_hist[0][1] - e_hist[0][0])/2)
plt.xlabel("Edge Length")
plt.ylabel("Edge Count")
plt.show()