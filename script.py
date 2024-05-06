if __name__ == '__main__':
    print('Starting script')
    import os
    from random import choice, randint

    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch import nn, optim
    from pickle import dump

    from tqdm import tqdm
    import networkx as nx
    import torch

    from pickle import load
    import torch.nn.functional as F

    import numpy as np
    from scipy.optimize import milp, LinearConstraint

    # torch geometric
    try:
        import torch_geometric
    except ModuleNotFoundError:
        print('Installing torch_geometric')
        import subprocess

        # Installing torch geometric packages with specific CUDA+PyTorch version.
        # See https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for details
        if torch.version.cuda is not None:
            TORCH = torch.__version__.split('+')[0]
            CUDA = 'cu' + torch.version.cuda.replace('.','')
            packages = ['torch-scatter', 'torch-sparse', 'torch-cluster', 'torch-spline-conv']
            for p in tqdm(packages, unit='pkg'):
                bashCommand = f"python3.8 -m pip install --no-index {p} -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html"
                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()

            # bashCommand = f"python3.8 -m pip install torch_geometric"
            # process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            # output, error = process.communicate()

        else:
            # bashCommand = f"pip install torch_geometric"
            # process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            # output, error = process.communicate()
            # print(f'{output}, {error}')

            bashCommand = f"pip install  torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html"
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

        # bashCommand = f"python3.8 -m pip install torch_geometric"
        # process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        # output, error = process.communicate()
        # print(f'{output}, {error}')

        print(f'{output=}, {error=}')
        import torch_geometric

    import torch_geometric.nn as geom_nn
    import torch_geometric.data as geom_data

    graphs_to_generate = 10000
    p = .15

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
        for i, edge in enumerate(zip(edge_index[0], edge_index[1])):
            v1, v2 = edge
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
        assert all(int(v1) in mvc or int(v2) in mvc
                   for v1, v2 in zip(edge_index[0], edge_index[1]))

        y = torch.LongTensor([int(n in mvc) for n in range(n)])
        x = torch.Tensor([[.1]] * 10)
        tg = geom_data.Data(x=x, y=y, edge_index=edge_index)

        pad_len = 20 - tg.edge_index.size(dim=1)
        tg.edge_index = F.pad(tg.edge_index, pad=(0, pad_len), value=0)
        graphs.append(tg)

    gnn_layer_by_name = {
        "GCN": geom_nn.GCNConv,
        "GAT": geom_nn.GATConv,
        "GraphConv": geom_nn.GraphConv
    }

    class GNNModel(nn.Module):

        def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN",
                     dp_rate=0.1, **kwargs):
            """
            Inputs:
                c_in - Dimension of input features
                c_hidden - Dimension of hidden features
                c_out - Dimension of the output features. Usually number of classes in classification
                num_layers - Number of "hidden" graph layers
                layer_name - String of the graph layer to use
                dp_rate - Dropout rate to apply throughout the network
                kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
            """
            super().__init__()
            gnn_layer = gnn_layer_by_name[layer_name]

            layers = []
            in_channels, out_channels = c_in, c_hidden
            for l_idx in range(num_layers - 1):
                layers += [
                    gnn_layer(in_channels=in_channels,
                              out_channels=out_channels,
                              **kwargs),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dp_rate)
                ]
                in_channels = c_hidden
            layers += [gnn_layer(in_channels=in_channels,
                                 out_channels=c_out,
                                 **kwargs)]
            self.layers = nn.ModuleList(layers)

        def forward(self, x, edge_index):
            """
            Inputs:
                x - Input features per node
                edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            """
            for l in self.layers:
                # For graph layers, we need to add the "edge_index" tensor as additional input
                # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
                # we can simply check the class type.
                if isinstance(l, geom_nn.MessagePassing):
                    x = l(x, edge_index)
                else:
                    x = l(x)
            return x


    class MLPModel(nn.Module):

        def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.1):
            """
            Inputs:
                c_in - Dimension of input features
                c_hidden - Dimension of hidden features
                c_out - Dimension of the output features. Usually number of classes in classification
                num_layers - Number of hidden layers
                dp_rate - Dropout rate to apply throughout the network
            """
            super().__init__()
            layers = []
            in_channels, out_channels = c_in, c_hidden
            for l_idx in range(num_layers - 1):
                layers += [
                    nn.Linear(in_channels, out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dp_rate)
                ]
                in_channels = c_hidden
            layers += [nn.Linear(in_channels, c_out)]
            self.layers = nn.Sequential(*layers)

        def forward(self, x, *args, **kwargs):
            """
            Inputs:
                x - Input features per node
            """
            return self.layers(x)


    import pytorch_lightning as pl

    CHECKPOINT_PATH = "./saved_models/tutorial7"

    class NodeLevelGNN(pl.LightningModule):

        def __init__(self, model_name, **model_kwargs):
            super().__init__()
            # Saving hyperparameters
            self.save_hyperparameters()

            if model_name == "MLP":
                self.model = MLPModel(**model_kwargs)
            else:
                self.model = GNNModel(**model_kwargs)
            self.loss_module = nn.CrossEntropyLoss()

        def forward(self, data, mode="train"):
            x, edge_index = data.x, data.edge_index
            x = self.model(x, edge_index)

            # Only calculate the loss on the nodes corresponding to the mask
            # if mode == "train":
            #     mask = data.train_mask
            # elif mode == "val":
            #     mask = data.val_mask
            # elif mode == "test":
            #     mask = data.test_mask
            # else:
            #     assert False, f"Unknown forward mode: {mode}"

            loss = self.loss_module(x, data.y)
            acc = (x.argmax(dim=-1) == data.y).sum().float() / data.y.size(dim=0)
            return loss, acc

        def configure_optimizers(self):
            # We use SGD here, but Adam works as well
            optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9,
                                  weight_decay=2e-3)
            return optimizer

        def training_step(self, batch, batch_idx):
            loss, acc = self.forward(batch, mode="train")
            self.log('train_loss', loss)
            self.log('train_acc', acc)
            return loss

        def validation_step(self, batch, batch_idx):
            _, acc = self.forward(batch, mode="val")
            self.log('val_acc', acc)

        def test_step(self, batch, batch_idx):
            _, acc = self.forward(batch, mode="test")
            self.log('test_acc', acc)


    def train_node_classifier(model_name, dataset, *, max_epochs=100, **model_kwargs):
        pl.seed_everything(42)
        end_train_i = int(len(dataset) * .7)
        end_val_i = int(len(dataset) * .9)
        train_data_loader = torch_geometric.data.DataLoader(
            dataset[:end_train_i], batch_size=1, shuffle=True, num_workers=7)
        val_data_loader = torch_geometric.data.DataLoader(
            dataset[end_train_i:end_val_i], batch_size=1, num_workers=7)
        test_data_loader = torch_geometric.data.DataLoader(
            dataset[end_val_i:], batch_size=1, num_workers=7)

        # Create a PyTorch Lightning trainer with the generation callback
        root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
        os.makedirs(root_dir, exist_ok=True)
        trainer = pl.Trainer(default_root_dir=root_dir,
                             callbacks=[
                                 ModelCheckpoint(save_weights_only=True, mode="max",
                                                 monitor="val_acc")],
                             # accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                             accelerator='gpu',
                             devices=1,
                             max_epochs=max_epochs,
                             enable_progress_bar=True)  # False because epoch size is 1
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filename = os.path.join(CHECKPOINT_PATH, f"NodeLevel{model_name}.ckpt")
        if os.path.isfile(pretrained_filename):
            print("Found pretrained model, loading...")
            model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
        else:
            pl.seed_everything()
            model = NodeLevelGNN(model_name=model_name, c_in=1, c_out=10, **model_kwargs)
            trainer.fit(model, train_data_loader, val_data_loader)
            model = NodeLevelGNN.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path)

        # Test best model on the test set
        test_result = trainer.test(model, test_data_loader, verbose=False)
        batch_t = next(iter(train_data_loader))
        batch_t = batch_t.to(model.device)
        _, train_acc = model.forward(batch_t)
        batch_v = next(iter(val_data_loader))
        batch_v = batch_v.to(model.device)
        _, val_acc = model.forward(batch_v)
        result = {"train": train_acc,
                  "val": val_acc,
                  "test": test_result[0]['test_acc']}
        return model, result


    # Small function for printing the test scores
    def print_results(result_dict):
        if "train" in result_dict:
            print(f"Train accuracy: {(100.0 * result_dict['train']):4.2f}%")
        if "val" in result_dict:
            print(f"Val accuracy:   {(100.0 * result_dict['val']):4.2f}%")
        print(f"Test accuracy:  {(100.0 * result_dict['test']):4.2f}%")


    node_mlp_model, node_mlp_result = train_node_classifier(model_name="MLP",
                                                            dataset=graphs,
                                                            c_hidden=16,
                                                            num_layers=2,
                                                            dp_rate=0.1)

    print_results(node_mlp_result)

    node_gnn_model, node_gnn_result = train_node_classifier(model_name="GNN",
                                                            layer_name="GCN",
                                                            dataset=graphs,
                                                            c_hidden=20,
                                                            num_layers=2,
                                                            dp_rate=0.1,
                                                            node_dim=0,
                                                            add_self_loops=False)
    print_results(node_gnn_result)
