import matplotlib.pyplot as plt
import torch

import dqc


def draw_graph(edge_list, nodes, edge_alpha=0.1, labels: bool = False):
    drawn_edges = torch.stack(
        [torch.stack([nodes[i.item()], nodes[j.item()]]).T for i, j in edge_list]
    )

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for e1, e2, e3 in drawn_edges:
        ax.plot(e1, e2, e3, color="black", alpha=edge_alpha)

    ax.scatter(*nodes.T, c="b", s=5)
    if labels:
        for i, _ in enumerate(nodes):
            ax.text(*nodes[i], f"{i}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def edge_list_to_adjacency_matrix(edge_list, num_nodes):
    # Initialize an empty adjacency matrix
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.int64)

    # Iterate through the edge list and populate the adjacency matrix
    for edge in edge_list:
        i, j = edge
        adjacency_matrix[i, j] += 1
        adjacency_matrix[j, i] += 1  # Assuming the graph is undirected

    return adjacency_matrix


if __name__ == "__main__":
    mol = dqc.Mol(
        "H 0.0 0.0 0.0; O 0.0 0.0 1.0",
        "pc-2",
        grid="sg2",
        graph="grid_neighbours",
        graph_kwargs={"range_modifier": 0.5, "expander_degree": 16},
    )
    mol.setup_grid()
    grid = mol.get_grid()
    graph = mol.get_graph()

    embedding = mol.get_embedding()

    plt.plot(embedding.atom_zs)
    plt.show()

    plt.plot(embedding._radial_dists)
    plt.show()

    adj_mat = edge_list_to_adjacency_matrix(graph, grid.get_rgrid().shape[0])

    print(graph.shape[0] / grid.get_rgrid().shape[0])

    plt.imshow(adj_mat)
    plt.colorbar()
    plt.show()
