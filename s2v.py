import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import MessagePassing


class Structure2VecConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Structure2Vec uses the sum of neighbor states
        super().__init__(aggr='add')

        # W1: Transforms fixed node features
        self.lin_node = Linear(in_channels, out_channels)
        # W2: Transforms aggregated neighbor hidden states
        self.lin_msg = Linear(out_channels, out_channels, bias=False)
        self.act = ReLU()

    def forward(self, x, edge_index, mu):
        # 1. Gather hidden states (mu) from neighbors
        msg = self.propagate(edge_index, mu=mu)

        # 2. Transform the summed messages (W2 * sum)
        aggregated = self.lin_msg(msg)

        # 3. Combine with transformed node features (W1 * x) and apply activation
        out = self.act(self.lin_node(x) + aggregated)
        return out

    def message(self, mu_j):
        # Pass the neighbor's hidden state unmodified (transformation happens after aggregation)
        return mu_j


class Structure2Vec(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_iterations=4):
        super().__init__()
        self.num_iterations = num_iterations
        self.hidden_channels = hidden_channels

        # We only instantiate ONE convolutional layer to share weights across all iterations
        self.conv = Structure2VecConv(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        # Initialize hidden states (mu) to zero for all nodes at t=0
        mu = torch.zeros((x.size(0), self.hidden_channels), device=x.device)

        # Iteratively update mu using the same shared weights
        for _ in range(self.num_iterations):
            mu = self.conv(x, edge_index, mu)

        # The final node embeddings after T iterations
        return mu