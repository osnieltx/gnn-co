from random import randint
from typing import Tuple

import numpy as np
import torch
from scipy.optimize import LinearConstraint, milp

from pyg import torch_geometric


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


def milp_solve_mds(edge_index, n):
    c = np.ones(n)
    A = np.identity(n)
    for v1, v2 in edge_index.T:
        A[v1, v2] = 1
        A[v2, v1] = 1

    b_l = np.ones(n)
    b_u = np.full_like(b_l, np.inf)

    constraints = LinearConstraint(A, b_l, b_u)
    integrality = np.ones_like(c)

    res = milp(c=c, constraints=constraints, integrality=integrality)
    mvc = {i for i, v in enumerate(res.x) if v}
    return mvc


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
