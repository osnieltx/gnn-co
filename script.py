
if __name__ == '__main__':
    print('Starting script')
    from random import randint

    import networkx as nx
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.optimize import milp, LinearConstraint
    from tqdm import tqdm

    from pyg import torch_geometric, geom_data
    from models import train_node_classifier

    graphs_to_generate = 10000
    p = .15
    torch.multiprocessing.set_sharing_strategy('file_system')

    def get_lonely_vertex(g, n):
        degress = [0] * n
        for v in g[0]:
            degress[v] += 1

        try:
            return degress.index(0)
        except ValueError:
            return None


    def create_graph(n):
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

    graphs = []
    for g_i in tqdm(range(graphs_to_generate), unit='graph'):
        n = 10
        edge_index = create_graph(n)

        mvc = milp_solve(edge_index, n)

        # check if valid vertex cover
        assert all(int(v1) in mvc or int(v2) in mvc for v1, v2 in edge_index.T)

        y = torch.FloatTensor([[n in mvc] for n in range(n)])
        x = torch.Tensor([[1.]] * 10)
        tg = geom_data.Data(x=x, y=y, edge_index=edge_index)

        pad_len = 20 - tg.edge_index.size(dim=1)
        tg.edge_index = F.pad(tg.edge_index, pad=(0, pad_len), value=0)
        graphs.append(tg)



#     node_mlp_model, node_mlp_result = train_node_classifier(model_name="MLP",
#                                                             dataset=graphs,
#                                                             c_hidden=16,
#                                                             num_layers=2,
#                                                             dp_rate=0.1)
#     print_results(node_mlp_result)

    node_gnn_model, node_gnn_result = train_node_classifier(
        model_name="GNN",
        layer_name="GCN",
        dataset=graphs,
        c_hidden=30,
        num_layers=20,
        node_dim=0,
        m=8,
        max_epochs=300
        # add_self_loops=False
    )
