from collections import OrderedDict, deque, namedtuple
from random import choice
from typing import Iterator, List, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch_geometric.data import Data, DataLoader
from torch.utils.data.dataset import IterableDataset

from graph import mds_is_solved, generate_graphs, milp_solve_mds
from pyg import geom_nn


gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}


class DQGN(nn.Module):
    def __init__(self, c_in, c_hidden=32, c_out=1, num_layers=20,
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
        output_layers = [gnn_layer(in_channels=in_channels, out_channels=c_out,
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

        if len(self.output_layers) > 1:
            probability_maps = torch.stack([
                out_l(x, edge_index)
                for out_l in self.output_layers
            ])
        else:
            probability_maps = self.output_layers[0](x, edge_index)

        return probability_maps
    

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
            self, n: int, p: float, s: int, replay_buffer: ReplayBuffer
    ) -> None:
        """Base Agent class handling the interaction with the environment.

        Args:
            replay_buffer: replay buffer storing experiences

        """
        self.graphs = generate_graphs(n, p, s)
        self.replay_buffer = replay_buffer
        self.state: torch.Tensor = None
        self.reset()

    def reset(self) -> None:
        """Resets the environment and updates the state."""
        self.state = choice(self.graphs)

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
        if np.random.random() < epsilon:
            action = choice(range(len(self.state.nx)))
        else:
            edge_index, node_feats = self.state.edge_index, self.state.x

            if device not in ["cpu"]:
                edge_index = edge_index.cuda(device)
                node_feats = node_feats.cuda(device)

            q_values = net(node_feats, edge_index).squeeze()
            # TODO validate
            breakpoint()
            current_solution = (self.state.x == .0)
            q_values[current_solution, 0] = float("-Inf")
            _, action = torch.max(q_values, dim=0)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
        episode_reward = 0,
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
        if self.state.x[action] == 0:
            return .0, False

        new_state = self.state.x.clone()
        new_state[action][0] = 0
        s = {i for i, x in enumerate(new_state) if x[0] == 0}
        solved = mds_is_solved(self.state.nx, s)

        reward = -1  # Negative reward for each additional node
        if solved:
            reward = len(new_state)  # Positive reward when domination achieved

        new_state_d = Data(new_state)
        exp = Experience(self.state.clone(), action, reward + episode_reward,
                         solved, new_state_d)

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
        episode_reward = 0,
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
        if self.state.x[action] == 0:
            return .0, False

        new_state = self.state.x.clone()
        new_state[action][0] = 0
        s = {i for i, x in enumerate(new_state) if x[0] == 0}
        solved = mds_is_solved(self.state.nx, s)

        reward = -1  # Negative reward for each additional node
        if solved:
            reward = len(new_state)  # Positive reward when domination achieved

        self.state.x = new_state
        return float(reward), solved

class DQNLightning(LightningModule):
    def __init__(
        self,
        n: int = 10,
        p: float = .15,
        s: int = 10000,
        batch_size: int = 500,
        lr: float = 1e-2,
        gamma: float = 0.99,
        sync_rate: int = 10,
        replay_size: int = 10000,
        eps_last_frame: int = 10000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 1000,
        warm_start_steps: int = 10000,
        validation_size: int = 10000,
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
        self.agent = Agent(n, p, s, self.buffer)

        model_kwargs['c_in'] = self.agent.state.x.size(dim=1)
        self.net = DQGN(**model_kwargs)
        self.target_net = DQGN(**model_kwargs)

        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially
         fill up the replay buffer with experiences.

        Args:
            steps: number of random steps to populate the buffer with

        """
        for _ in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

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
        epsilon = self.get_epsilon(self.hparams.eps_start, self.hparams.eps_end,
                                   self.hparams.eps_last_frame)
        self.log("epsilon", epsilon)

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device,
                                            self.episode_reward)
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
        self.log("total_reward", self.total_reward, prog_bar=True)
        self.log("steps", self.global_step, logger=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # TODO validate
        breakpoint()
        device = self.get_device(batch)
        total_reward = 0
        apx_ratio = 0
        for g in batch:
            episode_reward = 0
            self.agent.state = g
            while True:
                reward, done = self.agent.play_validation_step(
                    self.net, device=device, episode_reward=episode_reward)
                episode_reward += reward
                if done:
                    break
            total_reward += episode_reward
            apx_ratio += (self.agent.state.x == 0).sum(0) / (g.y == 1).sum(0)

        self.log("validation_avg_reward", total_reward/len(batch))
        self.log("apx_ratio_avg", apx_ratio/len(batch))

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving
        experiences."""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=7,
            persistent_workers=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def val_dataloder(self) -> DataLoader:
        graphs = generate_graphs(self.hparams.n, self.hparams.p,
                                 self.hparams.validation_size,
                                 solver=milp_solve_mds)
        val_data_loader = DataLoader(
            graphs, batch_size=self.hparams.batch_size, num_workers=7,
            persistent_workers=True)
        return val_data_loader

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0][0].x.device.index if self.on_gpu else "cpu"
