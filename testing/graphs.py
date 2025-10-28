import matplotlib.pyplot as plt
import torch
from dqc.grid.radial_grid import RadialGrid
from dqc.grid.lebedev_grid import LebedevGrid


def draw_graph(edge_list, nodes, edge_alpha=0.1, labels: bool = False):
    mask = torch.argwhere((nodes[:, 0] > 0))

    drawn_edges = torch.stack(
        [torch.stack([nodes[i.item()], nodes[j.item()]]).T for i, j in edge_list if i in mask and j in mask]
    )

    nodes = nodes[mask]


    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect([.5,1,1])

    for e1, e2, e3 in drawn_edges:
        ax.plot(e1, e2, e3, color="black", alpha=edge_alpha)

    ax.scatter(*nodes.T, c="b", s=5)
    if labels:
        for i, _ in enumerate(nodes):
            ax.text(*nodes[i], f"{i}")

    ax.set_axis_off()
    ax.grid(False)
    ax.set_xticks([]);
    ax.set_yticks([]);
    ax.set_zticks([])
    # Try to make 3D panes transparent (safe for older/newer Matplotlib)
    for a in (getattr(ax, "xaxis", None), getattr(ax, "yaxis", None), getattr(ax, "zaxis", None)):
        try:
            pane = getattr(a, "pane", None)
            if pane is not None:
                pane.set_alpha(0)
        except Exception:
            pass

    plt.show()


def draw_graph_2d(
    edge_list, nodes, edge_alpha=0.5, labels: bool = False, epsilon: float = 1e-6
):
    """Plot only nodes close to the X–Y plane (|z| <= epsilon) and edges between those nodes, in 2D."""
    # Select nodes that lie within the XY plane up to tolerance
    z = nodes[:, 2]
    x = nodes[:, 0]
    y = nodes[:, 1]
    mask = (z.abs() <= epsilon) & (x > 0) & (y > 0)

    # Early outcome: if no nodes satisfy the mask, show an empty 2D plot
    if mask.sum().item() == 0:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")
        plt.title("No nodes within the XY plane tolerance")
        plt.show()
        return

    # Nodes in the plane (projected to XY)
    plane_nodes = nodes[mask][:, :2]  # (K, 2)

    # Build mapping from original node indices -> new indices in plane_nodes
    N = nodes.shape[0]
    orig_indices = torch.arange(N, dtype=torch.long)[mask]
    idx_map = torch.full((N,), -1, dtype=torch.long)
    idx_map[orig_indices] = torch.arange(orig_indices.numel(), dtype=torch.long)

    # Normalize edge_list to a tensor of shape (E, 2)
    if isinstance(edge_list, torch.Tensor):
        edges = edge_list.to(dtype=torch.long)
    else:
        edges = torch.as_tensor(edge_list, dtype=torch.long)

    if edges.numel() == 0:
        valid_edges_new = torch.empty((0, 2), dtype=torch.long)
    else:
        edges = edges.view(-1, 2)
        # Map endpoints to the new compact index space
        mapped = idx_map[edges]
        # Keep only edges where both endpoints are inside the plane (mapping >= 0)
        valid_mask = (mapped[:, 0] >= 0) & (mapped[:, 1] >= 0)
        valid_edges_new = mapped[valid_mask]

    # Prepare 2D plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()

    # Draw edges (in 2D)
    if valid_edges_new.shape[0] > 0:
        xs = torch.stack(
            [
                plane_nodes[valid_edges_new[:, 0], 0],
                plane_nodes[valid_edges_new[:, 1], 0],
            ],
            dim=1,
        )
        ys = torch.stack(
            [
                plane_nodes[valid_edges_new[:, 0], 1],
                plane_nodes[valid_edges_new[:, 1], 1],
            ],
            dim=1,
        )
        for xseg, yseg in zip(xs, ys):
            ax.plot(xseg.tolist(), yseg.tolist(), color="black", alpha=edge_alpha)

    # Draw nodes (in 2D)
    ax.scatter(plane_nodes[:, 0].tolist(), plane_nodes[:, 1].tolist(), c="b", s=5)

    # Optional labels (labels indices correspond to the compact plane_nodes indexing)
    if labels:
        for i, (x, y) in enumerate(plane_nodes.tolist()):
            ax.text(x, y, f"{i}")

    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    ax.set_axis_off()
    ax.grid(False)
    ax.set_xticks([]);
    ax.set_yticks([]);
    ax.set_aspect("equal", adjustable="box")
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
    rad_grid = RadialGrid(10, grid_transform="none", grid_integrator="uniform_positive")

    grid = LebedevGrid(rad_grid, 41)

    graph = grid.generate_graph(
        graph_method="grid_neighbours",
        sparse=True,
        range_modifier=2.5,
    )

    rgrid = grid.get_rgrid()

    draw_graph_2d(graph, rgrid, edge_alpha=0.5, labels=False)
    plt.tight_layout()

    # c_vals = torch.arange(0, 2, 0.01)
    #
    # degree = []
    # for c in c_vals:
    #     grid = get_predefined_grid(
    #         "sg2", [1], torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    #     )
    #     graph = grid.generate_graph(
    #         graph_method="grid_neighbours",
    #         sparse=True,
    #         range_modifier=c.item(),
    #     )
    #     degree.append(graph.shape[0] / grid.get_rgrid().shape[0])
    #
    # plt.plot(c_vals.numpy(), degree)
    # plt.ylabel("Average Degree")
    # plt.xlabel("Range Modifier")
    # plt.grid()
    # plt.ylim(1, None)
    # plt.xlim(-1e-2, 2)
    # plt.show()

    # mol = dqc.Mol(
    #     "H 0.0 0.0 0.0; O 0.0 0.0 1.0",
    #     "pc-2",
    #     grid="sg2",
    #     graph="grid_neighbours",
    #     graph_kwargs={"range_modifier": 0.5, "expander_degree": 16},
    # )
    # mol.setup_grid()
    # grid = mol.get_grid()
    # graph = mol.get_graph()
    #
    # embedding = mol.get_embedding()
    #
    # plt.plot(embedding._atom_zs)
    # plt.show()
    #
    # plt.plot(embedding._radial_dists)
    # plt.show()
    #
    # adj_mat = edge_list_to_adjacency_matrix(graph, grid.get_rgrid().shape[0])
    #
    # print(graph.shape[0] / grid.get_rgrid().shape[0])
    #
    # plt.imshow(adj_mat)
    # plt.colorbar()
    # plt.show()
