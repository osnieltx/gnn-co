from random import randint

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
        if not v:
            break

        u = randint(0, n-1)
        if u == v:
            continue
        new_edge = torch.tensor([[v, u], [u, v]])
        ei = torch.cat((ei, new_edge), 1)

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
