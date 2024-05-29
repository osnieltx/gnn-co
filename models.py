import os
from collections import namedtuple
from datetime import datetime
from functools import partial

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import KFold
from torch import nn, optim

from pyg import torch_geometric, geom_nn

sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}

class GNNModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN",
                 dp_rate=0.1, m=4, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            m - Number of output maps, int
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
                # nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        self.layers = nn.ModuleList(layers)
        output_layers = [gnn_layer(in_channels=in_channels,
                                   out_channels=c_out,
                                   **kwargs)
                         for _ in range(m)]
        self.output_layers = nn.ModuleList(output_layers)


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

        probability_maps = torch.stack([
            out_l(x, edge_index)
            for out_l in self.output_layers
        ])

        return probability_maps


class MLPModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.0):
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


NodeFowardResult = namedtuple('NodeFowardResult',
                              ['loss', 'acc', 'aon', 'mvc_score'])

class NodeLevelGNN(pl.LightningModule):
    def __init__(self, model_name, batch_size=None, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        else:
            self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.BCEWithLogitsLoss()

        self.log = partial(self.log, batch_size=batch_size)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        prob_maps = self.model(x, edge_index)

        loss = min(self.loss_module(pb, data.y) for pb in prob_maps)

        maps = (torch.sigmoid(prob_maps) > .5).float()

        acc = (maps == data.y).sum(dim=1).float() / data.y.size(dim=0)
        acc = acc.max()

        aon = (maps == data.y).all(dim=1).float().mean()

        a, b = 1, 1
        cov_size_dif = (maps.sum(dim=1) - data.y.sum()).abs()
        uncovered_edges = torch.sum(
            ~(maps[:, edge_index[0]].logical_or(maps[:, edge_index[1]])),
            dim=1
        )
        mvc_score = (a * cov_size_dif + b * uncovered_edges).min()

        return NodeFowardResult(loss, acc, aon, mvc_score)

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def training_step(self, batch, batch_idx):
        result = self.forward(batch)
        self.log('train_loss', result.loss)
        self.log('train_acc', result.acc)
        self.log('train_aon', result.aon)
        self.log('train_mvc_s', result.mvc_score)
        return result.loss

    def validation_step(self, batch, batch_idx):
        result = self.forward(batch)
        self.log('val_loss', result.loss)
        self.log('val_acc', result.acc)
        self.log('val_aon', result.aon)
        self.log('val_mvc_s', result.mvc_score)

    def test_step(self, batch, batch_idx):
        result = self.forward(batch)
        self.log('test_acc', result.acc)
        self.log('test_aon', result.aon)
        self.log('test_mvc_s', result.mvc_score)


def train_node_classifier(model_name, dataset, *, max_epochs=100, **model_kwargs):
    pl.seed_everything(42)

    models = []
    results = []
    kf = KFold()
    date = str(datetime.now())[:16]
    date = date.replace(':', '')
    model_dir = f'experiments/{date}'
    for i, (train_index, test_index) in enumerate(kf.split(dataset)):
        print(f'Training fold 🗂️  {i+1}/5')

        os.makedirs(model_dir, exist_ok=True)
        logger = CSVLogger('experiments/', name=date, version=str(i))
        batch_size = 1

        train_data_loader = torch_geometric.data.DataLoader(
            [g for gi, g in enumerate(dataset) if gi in train_index],
            batch_size=batch_size, shuffle=True, num_workers=7,
            persistent_workers=True,
            worker_init_fn=set_worker_sharing_strategy)
        val_data_loader = torch_geometric.data.DataLoader(
            [g for gi, g in enumerate(dataset) if gi in test_index],
            batch_size=batch_size, num_workers=7, persistent_workers=True,
            worker_init_fn=set_worker_sharing_strategy)

        # Create a PyTorch Lightning trainer with the generation callback
        trainer = pl.Trainer(callbacks=[
                                 ModelCheckpoint(save_weights_only=True,
                                                 mode="min",
                                                 monitor="val_loss"),
                                 EarlyStopping('val_loss', patience=50)],
                             accelerator='auto',
                             devices=1,
                             max_epochs=max_epochs,
                             enable_progress_bar=True,
                             logger=logger)  # False because epoch size is 1
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

        model = NodeLevelGNN(model_name=model_name, batch_size=batch_size,
                             c_in=1, c_out=1, **model_kwargs)
        trainer.fit(model, train_data_loader, val_data_loader)
        model = NodeLevelGNN.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)

        # Test best model on the test set
        test_result = trainer.test(model, val_data_loader, verbose=False)
        batch_t = next(iter(train_data_loader))
        batch_t = batch_t.to(model.device)
        t_result = model.forward(batch_t)
        batch_v = next(iter(val_data_loader))
        batch_v = batch_v.to(model.device)
        v_result = model.forward(batch_v)
        result = {"train": t_result.acc,
                  "val": v_result.acc,
                  "test": test_result[0]['test_acc']}

        print_results(result)

        models.append(model)
        results.append(result)
    return models, results


# Small function for printing the test scores
def print_results(result_dicts):
    if not isinstance(result_dicts, list):
        result_dicts = [result_dicts]

    for i, result in enumerate(result_dicts):
        if len(result_dicts) > 1:
            print(f'Model {i}:')

        if "train" in result:
            print(f"Train accuracy: {(100.0 * result['train']):4.2f}%")
        if "val" in result:
            print(f"Val accuracy:   {(100.0 * result['val']):4.2f}%")

        print(f"Test accuracy:  {(100.0 * result['test']):4.2f}%")

