from random import randint
from typing import Tuple

import numpy as np
import torch
from scipy.optimize import LinearConstraint, milp

from pyg import torch_geometric, geom_data


def get_lonely_vertex(g, n):
    degress = [0] * n
    for v in g[0]:
        degress[v] += 1

    try:
        return degress.index(0)
    except ValueError:
        return None


def create_graph(n, p=.15):
    ei = torch_geometric.utils.erdos_renyi_graph(n, p)

    while True:
        v = get_lonely_vertex(ei, n)
        if v is None:
            break

        u = randint(0, n-1)
        if u == v:
            continue
        new_edge = torch.tensor([[v, u], [u, v]])
        ei = torch.cat((ei, new_edge), 1)

    assert ei[1].unique().size(0) == n

    return ei


def prepare_graph(i, n, p, solver, dataset_dir=None):
    edge_index = create_graph(n, p)
    s = solver(edge_index, n, time_limit=120)
    y = torch.FloatTensor([[n in s] for n in range(n)])
    x = torch.FloatTensor([[1]] * n)
    # x = clustering_coefficient(edge_index)[:, 1]
    # d_g = x.max().item()
    # if d_g > max_d:
    #     max_d = d_g
    g = geom_data.Data(x=x, y=y, edge_index=edge_index)
    if dataset_dir:
        torch.save(g, f'{dataset_dir}/{i}.pt')
    return g


def clustering_coefficient(g: torch.Tensor, verbose=False) -> torch.Tensor:
    """
    Given the incidence matrix g, computes the clustering coefficient for each
    node.
    :param g: An incidence matrix of shape (2, |E|), where each column
     represents an edge by specifying the two nodes it connects.
    :param verbose: If True prints some debugging information.
    :return: A tensor of shape (n, 2), where the first column is the number of
     neighbors and the second column is the clustering coefficient.
    """
    num_edges = g.shape[1]
    num_nodes = g.max().item() + 1  # Assuming nodes are labeled from 0 to n-1
    results = torch.zeros((num_nodes, 2))

    # Construct adjacency matrix from incidence matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.int)
    for edge in range(num_edges):
        u, v = g[:, edge]
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1

    if verbose: print(adj_matrix)

    for v in range(num_nodes):
        neighbors = adj_matrix[v].nonzero(as_tuple=False).squeeze()
        if verbose: print(f'{neighbors=}')
        if neighbors.ndimension() == 0:
            neighbors = neighbors.unsqueeze(0)
        k_v = neighbors.size(0)

        # Store number of neighbors
        results[v, 0] = k_v

        if k_v < 2:
            # Clustering coefficient is undefined for
            # nodes with less than two neighbors
            continue

        # Subgraph induced by neighbors
        subgraph = adj_matrix[neighbors][:, neighbors]
        # Each edge is counted twice in the adjacency matrix
        actual_edges = torch.sum(subgraph) / 2
        possible_edges = k_v * (k_v - 1) / 2

        results[v, 1] = actual_edges / possible_edges

    return results


def jaccard_coefficient(g: torch.Tensor, n, max_d) -> torch.Tensor:
    g = g.T.tolist()
    # Create adjacency set from the edge index matrix
    neig = {i: set() for i in range(n)}
    for u, v in g:
        neig[u].add(v)
        neig[v].add(u)

    jac_coefs = torch.zeros((n, int(max_d)))
    for u in range(n):
        for i, v in enumerate(neig[u]):
            nu_or_nv = len((neig[u] | neig[v]) - {u, v})
            if not nu_or_nv:
                continue
            coef = len(neig[u] & neig[v]) / nu_or_nv
            jac_coefs[u, i] = coef

    return jac_coefs


# ---------------  MILP SOLVERS ---------------------------------------


def milp_solve(edge_index, n):
    # Solving MVC with MILP
    c = np.ones(n)
    A = np.zeros((len(edge_index[0]), n))
    for i, (v1, v2) in enumerate(edge_index.T):
        A[i, v1] = 1
        A[i, v2] = 1

    b_l = np.ones(len(edge_index[0]))
    b_u = np.full_like(b_l, np.inf)

    constraints = LinearConstraint(A, b_l, b_u)
    integrality = np.ones_like(c)

    res = milp(c=c, constraints=constraints, integrality=integrality)
    mvc = {i for i, v in enumerate(res.x) if v}
    return mvc


def milp_solve_mds(edge_index, n, **options):
    c = np.ones(n)
    A = np.identity(n)
    for v1, v2 in edge_index.T:
        A[v1, v2] = 1
        A[v2, v1] = 1

    b_l = np.ones(n)
    b_u = np.full_like(b_l, np.inf)

    constraints = LinearConstraint(A, b_l, b_u)
    integrality = np.ones_like(c)

    res = milp(c=c, constraints=constraints, integrality=integrality,
               options=options)
    mvc = {i for i, v in enumerate(res.x) if v}
    return mvc

# ---------------  PROBLEM SCORERS ---------------------------------------


def mdsi(maps, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    :return:
    cov_size, uncovered_nodes, best_sol
    """
    cov_size = maps.sum(dim=1)

    not_in_s = (maps == 0)
    has_no_neighbors_in_s = torch.ones_like(maps, dtype=torch.bool)
    u_in_s = maps[:, g[0]] == 1
    v_in_s = maps[:, g[1]] == 1
    has_no_neighbors_in_s[:, g[0]] &= ~v_in_s
    has_no_neighbors_in_s[:, g[1]] &= ~u_in_s
    uncovered_nodes = (not_in_s & has_no_neighbors_in_s).sum(dim=1)

    mds_scores = cov_size + uncovered_nodes

    return cov_size, uncovered_nodes, mds_scores.argmin().item()


def mvci(maps, g: torch.Tensor) -> int:
    cov_size = maps.sum(dim=1)
    uncovered_edges = torch.sum(
        ~(maps[:, g[0]].logical_or(maps[:, g[1]])),
        dim=1
    ) / 2
    return (cov_size + uncovered_edges).argmin()
