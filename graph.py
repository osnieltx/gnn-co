from copy import copy
from functools import partial
from multiprocessing import Pool
from random import randint, choice
from typing import Tuple

from pyg import torch_geometric, geom_data
from scipy.optimize import LinearConstraint, milp
from tqdm import tqdm
import gurobipy as gp
import networkx as nx
import numpy as np
import torch

from gurobi_manager import options


# ---------------  GRAPH MANIPULATIONS ---------------------------------------


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


def load_graph(g_id, path):
    return torch.load(f'{path}/{g_id}.pt')


def prepare_graph(i, n_r: range, p, solver=None, dataset_dir=None,
                  g_nx=False, solver_kwargs=None, attrs=None):
    attrs = attrs if attrs is not None else []
    n = choice(n_r)
    edge_index = create_graph(n, p)
    if solver:
        s = solver(edge_index, n, **(solver_kwargs or {}))
        y = torch.FloatTensor([[n in s] for n in range(n)])
    else:
        y = None
    x = torch.FloatTensor([[.0]] * n)
    if 'dominable_neighbors' in attrs:
        dns = dominable_neighbors(edge_index).unsqueeze(1)
        x = torch.cat((x, dns), 1)

    # x = clustering_coefficient(edge_index)[:, 1]
    # d_g = x.max().item()
    # if d_g > max_d:
    #     max_d = d_g
    g_nx = nx.from_edgelist(edge_index.T.tolist()) if g_nx else None
    g = geom_data.Data(x=x, y=y, edge_index=edge_index, nx=g_nx)
    if dataset_dir:
        torch.save(g, f'{dataset_dir}/{i}.pt')
    return g


def generate_graphs(n_r: range, p, s, solver=None, dataset_dir=None,
                    attrs=None):
    print(f'Sampling {s} instances from G({n_r}, {p})...')
    with Pool() as pool:
        get_graph = partial(prepare_graph, n_r=n_r, p=p, g_nx=True,
                            solver=solver, dataset_dir=dataset_dir,
                            attrs=attrs)
        return list(tqdm(
            pool.imap_unordered(get_graph, range(s)), total=s, unit='graph')
        )


# ---------------  ATTRIBUTES ---------------------------------------

def dominable_neighbors(g: torch.Tensor, s = None):
    if s is None:
        s = set()

    nodes, degress = torch.unique(g[0], return_counts=True)
    dominable = degress + 1

    d = copy(s)
    for u, v in g.T.tolist():
        if u in s:
            d.add(v)
        if v in s:
            d.add(u)

    for u, v in g.T.tolist():
        if u in d:
            dominable[v] -= 1

    for i in s:
        dominable[i] = 0

    return dominable/(degress.max()+1)


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
    with gp.Env(params=options) as env, gp.Model(env=env) as m:
        m.Params.TimeLimit = 1 * 60 * 60

        c = np.ones(n)
        x = m.addMVar(shape=n, vtype=gp.GRB.BINARY, name="x")
        A = np.zeros((len(edge_index[0]), n))  # incidence matrix
        for i, (v1, v2) in enumerate(edge_index.T):
            A[i, v1] = 1
            A[i, v2] = 1

        b_l = np.ones(len(edge_index[0]))
        b_u = np.full_like(b_l, np.inf)

        m.addConstr(A @ x >= b_l, name="cl")
        m.addConstr(A @ x <= b_u, name="cu")

        m.setObjective(c @ x, gp.GRB.MINIMIZE)
        m.optimize()

        mvc = {i for i, v in enumerate(x.X) if v}
        return mvc


def milp_solve_mds(edge_index, n, **options):
    with gp.Env(params=options) as env, gp.Model(env=env) as m:
        m.Params.TimeLimit = 1 * 60 * 60
        m.Params.OutputFlag = 0

        c = np.ones(n)
        x = m.addMVar(shape=n, vtype=gp.GRB.BINARY, name="x")
        A = np.identity(n)  # adj. matrix
        for v1, v2 in edge_index.T:
            A[v1, v2] = 1
            A[v2, v1] = 1

        b_l = np.ones(n)
        b_u = np.full_like(b_l, np.inf)

        m.addConstr(A @ x >= b_l, name="lc")
        m.addConstr(A @ x <= b_u, name="uc")

        m.setObjective(c @ x, gp.GRB.MINIMIZE)
        m.optimize()
        mvc = {i for i, v in enumerate(x.X) if v}
        return mvc


def is_ds(g, s: set):
    """
    Checks if a set S ⊆ V(G) is a dominating set of a graph G.

    Parameters:
    g (nx.Graph): Graph G, where g[v] gives N(v).
    s (set): Subset S ⊆ V(G).

    Returns:
    bool: True if ∀v ∈ V, v ∈ S or ∃u ∈ N(v) ∩ S. False otherwise.
    """
    return all(v in s or any(n in s
                             for n in g[v])
               for v in g)

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
