from graph import create_graph, milp_solve

if __name__ == '__main__':
    print('Starting script')
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    from pyg import geom_data
    from models import train_node_classifier

    graphs_to_generate = 10000
    torch.multiprocessing.set_sharing_strategy('file_system')

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

        # pad_len = 20 - tg.edge_index.size(dim=1)
        # tg.edge_index = F.pad(tg.edge_index, pad=(0, pad_len), value=0)
        graphs.append(tg)

    node_gnn_model, node_gnn_result = train_node_classifier(
        model_name="GNN",
        layer_name="GCN",
        dataset=graphs,
        c_hidden=30,
        num_layers=20,
        node_dim=0,
        m=16,
        max_epochs=300
        # add_self_loops=False
    )
