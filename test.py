import torch

from graph import create_graph, milp_solve
from models import NodeLevelGNN
from pyg import geom_data

n = 10
g = create_graph(n)
print(f'created graph with edges:\n{g.T}')
x = torch.Tensor([[1.]] * n)
x = geom_data.Data(x=x, edge_index=g)

model = NodeLevelGNN.load_from_checkpoint("/path/to/checkpoint.ckpt")
model.eval()  # disable randomness, dropout, etc...

prob_maps = model(x)
maps = (torch.sigmoid(prob_maps) > .5).float()
print(f'model outputs:\n{maps}')

y = milp_solve(g, n)
print(f'{y=}')
y = torch.FloatTensor([[n in y] for n in range(n)])

acc = (maps == y).sum(dim=1).float() / y.size(dim=0)
acc = acc.max()
print(f'{acc=}')

aon = (maps == y).all(dim=1).sum().float()
print(f'{aon=}')

a, b = 1, 1
cov_size_dif = (maps.sum(dim=1) - y.sum()).abs()
uncovered_edges = torch.sum(
    ~(maps[:, g[0]].logical_or(maps[:, g[1]])),
    dim=1
)
mvc_score = (a * cov_size_dif + b * uncovered_edges).min()
print(f'{mvc_score=}')