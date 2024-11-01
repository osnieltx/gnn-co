from collections import OrderedDict, deque, namedtuple
from functools import partial
from random import choice
from typing import Iterator, List, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_add_pool
from torch.utils.data.dataset import IterableDataset

from graph import mds_is_solved, generate_graphs, milp_solve_mds
from pyg import geom_nn


gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}


class DQGN(nn.Module):
    def __init__(self, c_in, c_hidden=64, c_out=1, num_layers=10,
                 layer_name="GCN", dp_rate=None, m=1, **kwargs):
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
                gnn_layer(in_channels=in_channels, out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
            ]
            if dp_rate:
                layers += [nn.Dropout(dp_rate)]
            in_channels = c_hidden
        self.layers = nn.ModuleList(layers)
        self.node_transform = nn.Linear(in_channels, in_channels,
                                        bias=False)
        self.neig_transform = nn.Linear(in_channels, in_channels,
                                        bias=False)
        self.aggr_transform = nn.Linear(2*in_channels, c_out,
                                        bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

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

        x = self.node_transform(x)
        nodes_pool_sum = x.sum(dim=0, keepdim=x.dim() <= 2)
        pool_transformed = self.neig_transform(nodes_pool_sum)
        repeated_pool = pool_transformed.repeat(x.shape[0], 1)
        x = torch.cat((x, repeated_pool), 1)
        x = self.relu(x)
        x = self.aggr_transform(x)
        x = self.tanh(x)

        return x
    

# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn
    from them.

    Args:
        capacity: size of the buffer

    """

    def __init__(self, capacity: int, action_space_size) -> None:
        self.buffer = deque(maxlen=capacity)
        self.a_size = action_space_size

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)

        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *(self.buffer[idx] for idx in indices))

        return (
            list(states),
            list(actions),
            list(rewards),
            list(dones),
            list(next_states),
        )


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time

    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        states, actions, rewards, dones, new_states = self.buffer.sample(
            self.sample_size)
        for i in range(len(dones)):
            yield (states[i], actions[i], rewards[i], dones[i],
                   new_states[i])


class Agent:
    def __init__(
        self, n: int, p: float, s: int, replay_buffer: ReplayBuffer,
        n_step: int
    ) -> None:
        """Base Agent class handling the interaction with the environment.

        Args:
            replay_buffer: replay buffer storing experiences

        """
        self.graphs = generate_graphs(n, p, s)
        self.replay_buffer = replay_buffer
        self.state: torch.Tensor = None
        self.n_step = n_step
        self.reset()

    def reset(self) -> None:
        """Resets the environment and updates the state."""
        self.state = choice(self.graphs).clone()
        self.state.step = 0
        self.state.history = []

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an
        epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action

        """
        current_solution = (self.state.x == 1).squeeze()
        if np.random.random() < epsilon:
            action = (~current_solution).float().multinomial(1)
        else:
            edge_index, node_feats = self.state.edge_index, self.state.x

            if device not in ["cpu"]:
                edge_index = edge_index.cuda(device)
                node_feats = node_feats.cuda(device)

            q_values = net(node_feats, edge_index).squeeze()
            q_values[current_solution] = float("-Inf")
            _, action = torch.max(q_values, dim=0)

        action = int(action.item())
        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done

        """
        action = self.get_action(net, epsilon, device)
        if self.state.x[action] == 1:
            return .0, False

        new_state = self.state.x.clone()
        new_state[action][0] = 1
        s = {i for i, x in enumerate(new_state) if x[0] == 1}
        solved = mds_is_solved(self.state.nx, s)

        reward = len(self.state.nx[action])
        reward = -1  # Negative reward for each additional node
        reward /= len(new_state)

        exp = Experience(self.state.clone(), action, reward, solved, None)
        self.state.history.append(exp)
        self.state.step += 1

        if self.state.step >= self.n_step:
            total_r = sum(s.reward for s in self.state.history)
            exp = self.state.history.pop(0)
            exp = exp._replace(new_state=Data(new_state), reward=total_r,
                               done=solved)
            del exp.state.history, exp.state.step
            self.replay_buffer.append(exp)

        self.state.x = new_state
        if solved:
            self.reset()
        return float(reward), solved

    @torch.no_grad()
    def play_validation_step(
        self,
        net: nn.Module,
        device: str = "cpu",
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done

        """
        action = self.get_action(net, 0, device)
        if self.state.x[action] == 1:
            return .0, False

        new_state = self.state.x.clone()
        new_state[action][0] = 1
        s = {i for i, x in enumerate(new_state) if x[0] == 1}
        solved = mds_is_solved(self.state.nx, s)
        reward = -1  # Negative reward for each additional node
        reward /= len(new_state)
        self.state.x = new_state
        return float(reward), solved

class DQNLightning(LightningModule):
    def __init__(
        self,
        n: int = 10,
        p: float = .15,
        s: int = 10000,
        batch_size: int = 5000,
        lr: float = 1e-4,
        gamma: float = 0.99,
        sync_rate: int = 10,
        replay_size: int = 10000,
        eps_last_frame: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        episode_length: int = 5000,
        warm_start_steps: int = 10000,
        validation_size: int = 1000,
        n_step: int = 2,
        **model_kwargs
    ) -> None:
        """Basic DQN Model.

        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            warm_start_steps: max episode reward in the environment

        """
        super().__init__()
        self.save_hyperparameters()

        self.buffer = ReplayBuffer(self.hparams.replay_size, n)
        self.agent = Agent(n, p, s, self.buffer, n_step)

        model_kwargs['c_in'] = self.agent.state.x.size(dim=1)
        self.net = DQGN(**model_kwargs)
        self.target_net = DQGN(**model_kwargs)

        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)
        self.log = partial(self.log, batch_size=batch_size)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially
         fill up the replay buffer with experiences.

        Args:
            steps: number of random steps to populate the buffer with

        """
        for _ in range(steps):
            step_r, done = self.agent.play_step(self.net, epsilon=1.0)


    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of
        each action as an output.

        Args:
            x: environment state

        Returns:
            q values

        """
        output = self.net(x, edge_index)
        return output

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss

        """

        states, actions, rewards, dones, next_states = batch
        state_action_values = self.net(states.x, states.edge_index).reshape(
           (self.hparams.batch_size, self.hparams.n)).gather(
               1, actions.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(
                next_states.x, states.edge_index
            )
            next_state_values = next_state_values.reshape(
                (self.hparams.batch_size, self.hparams.n)).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = (next_state_values * self.hparams.gamma
                                        + rewards)

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def training_step(
            self, batch: Tuple[Tensor, Tensor], nb_batch
    ) -> OrderedDict:
        """Carries out a single step through the environment to update the
        replay buffer. Then calculates loss based on the minibatch received.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics

        """
        device = self.get_device(batch)
        epsilon = self.get_epsilon(self.hparams.eps_start,
                                   self.hparams.eps_end,
                                   self.hparams.eps_last_frame)
        self.log("epsilon", epsilon)

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        if not reward:
            breakpoint()
        self.episode_reward += reward
        self.log("episode reward", self.episode_reward)

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict(
            {
                "reward": reward,
                "train_loss": loss,
            }
        )
        self.log("total_reward", float(self.total_reward), prog_bar=True)
        self.log("steps", float(self.global_step), logger=False, prog_bar=True)
        self.log("lr", self.lr_schedulers().get_last_lr()[0])

        return loss

    def validation_step(self, batch, batch_idx):
        # if self.global_step == 599:
        #     breakpoint()
        old_agent_state = self.agent.state
        device = self.get_device(batch)
        total_reward = 0
        val_apx_ratio = 0
        for g in batch.to_data_list():
            episode_reward = 0
            self.agent.state = g
            while True:
                reward, done = self.agent.play_validation_step(
                    self.net, device=device) 
                episode_reward += reward
                if done:
                    break
            total_reward += episode_reward
            sol_size = (self.agent.state.x == 1).sum(0)
            opt_size = (g.y == 1).sum(0)
            val_apx_ratio += sol_size / opt_size 

        self.log("val_avg_reward", total_reward/batch.num_graphs)
        self.log("val_apx_ratio", val_apx_ratio/batch.num_graphs)
        self.agent.state = old_agent_state

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=20)
        return {'optimizer': optimizer, "lr_scheduler": scheduler}

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving
        experiences."""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    # def val_dataloader(self) -> DataLoader:
    #     graphs = generate_graphs(self.hparams.n, self.hparams.p,
    #                              self.hparams.validation_size,
    #                              solver=milp_solve_mds)
    #     val_data_loader = DataLoader(
    #         graphs, batch_size=self.hparams.batch_size, num_workers=7,
    #         persistent_workers=True)
    #     return val_data_loader

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        try:
            return batch[0][0].x.device.index if self.on_gpu else "cpu"
        except:
            return batch[0].x.device.index if self.on_gpu else "cpu"

