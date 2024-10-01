import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch_geometric.utils as pyg_utils
from torch.distributions import Categorical
import torch_geometric.data as Data
from tqdm import trange

from graph import mds_is_solved

class RLAgent(torch.nn.Module):
    def __init__(self, c_in, c_hidden, c_out, heads=2):
        super(RLAgent, self).__init__()
        # Use GATConv layers with attention heads
        self.gat1 = GATConv(c_in, c_hidden, heads=heads, concat=True)
        self.gat2 = GATConv(c_hidden * heads, c_out, heads=1, concat=False)
        self.fc = torch.nn.Linear(c_out, 1)  # Output is 1 for MDS decision

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return torch.sigmoid(self.fc(x))  # Output probabilities for each node

class MDSRL:
    def __init__(self, gnn, lr=0.001, gamma=0.99):
        self.gnn = gnn
        self.optimizer = optim.Adam(gnn.parameters(), lr=lr)
        self.gamma = gamma
        self.eps = 0.1  # Epsilon-greedy for exploration
        self.memory = []

    def select_action(self, graph):
        edge_index, node_feats = graph.edge_index, graph.x
        prob = self.gnn(node_feats, edge_index).squeeze()
        m = Categorical(prob)
        action = m.sample()
        self.memory.append((m.log_prob(action), action))
        return action.item()

    def optimize(self, rewards):
        loss = []
        G = 0
        for (log_prob, action), reward in zip(reversed(self.memory), reversed(rewards)):
            G = reward + self.gamma * G
            loss.append(-log_prob * G)
        self.optimizer.zero_grad()
        total_loss = sum(loss)  # Use sum() for scalar tensors
        total_loss.backward()  # Backpropagate the loss
        self.optimizer.step()
        self.memory = []

def train_rl_agent(agent, graphs, n_epochs, n):
    for epoch in trange(n_epochs, unit='epoch'):
        for graph in graphs:
            state = graph  # Assume graph has node features `x` and `edge_index`
            actions = []
            rewards = []

            # Perform an episode of actions
            for step in range(len(state.x)):
                action = agent.select_action(state)
                actions.append(action)
                state.x[action][0] = 0

                # Update graph and calculate reward
                reward = -1  # Negative reward for each additional node
                if action in actions:
                    reward = -10
                if mds_is_solved(graph.nx, actions):
                    reward = n  # Positive reward when domination achieved
                rewards.append(reward)

            # Optimize the policy after each graph episode
            agent.optimize(rewards)
