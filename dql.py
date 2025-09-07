import concurrent
import threading
from collections import OrderedDict, deque, namedtuple
from functools import partial
from random import choice
from typing import Iterator, List, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.optim import Adam, Optimizer, lr_scheduler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool
from torch.utils.data.dataset import IterableDataset

from graph import generate_graphs
from pyg import geom_nn


gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}


class DQGN(nn.Module):
    def __init__(self, c_in, c_hidden=64, c_out=1,
                 num_layers=10, layer_name="GCN", dp_rate=None,
                 aggr_out_by_graph=False, **gnn_kwargs):
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

        self.aggr_out_by_graph = aggr_out_by_graph

        gnn_layer = gnn_layer_by_name[layer_name]
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels,
                          **gnn_kwargs),
                nn.ReLU(inplace=True),
            ]
            if dp_rate:
                layers += [nn.Dropout(dp_rate)]
            in_channels = c_hidden
        self.layers = nn.ModuleList(layers)
        self.node_transform = nn.Linear(in_channels, in_channels, bias=False)

        if self.aggr_out_by_graph:
            self.grph_transform = nn.Linear(in_channels, c_out, bias=False)
        else:
            self.grph_transform = nn.Linear(in_channels, in_channels,
                                            bias=False)
            self.aggr_transform = nn.Linear(2*in_channels, c_out, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x, edge_index, nb_batch):
        """
        Inputs: x - Input features per node edge_index - List of vertex index
        pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)

        x = self.node_transform(x)
        nodes_pool_sum = global_add_pool(x, nb_batch)
        pool_transformed = self.grph_transform(nodes_pool_sum)
        if self.aggr_out_by_graph:
            x = pool_transformed
        else:
            repeated_pool = pool_transformed[nb_batch]
            x = torch.cat((x, repeated_pool), 1)
            x = self.relu(x)
            x = self.aggr_transform(x)
        x = self.tanh(x)

        return x
    

# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state",
                 "total_reward"],
)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn
    from them.

    Args:
        capacity: size of the buffer

    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)

        """
        with self.lock:
            self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        total_episode_rewards = np.array([event.total_reward
                                          for event in self.buffer])
        weights = np.abs(total_episode_rewards)

        # Invert weights: smaller weights get higher sampling probabilities
        # Add epsilon to avoid division by zero
        inverse_weights = 1.0 / (weights + 1e-8)  
        probabilities = inverse_weights / np.sum(inverse_weights)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False,
                                   p=probabilities)

        # Gather samples based on indices
        states, actions, rewards, dones, next_states, _ = zip(
            *(self.buffer[idx] for idx in indices)
        )

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
        self, n_r: range, p: float, s: int, replay_buffer: ReplayBuffer,
        n_step: int, graphs=None, graph_attr_func=None, check_solved=None
    ) -> None:
        """Base Agent class handling the interaction with the environment.

        Args:
            replay_buffer: replay buffer storing experiences

        """
        self.graph_attr_func = graph_attr_func
        self.graphs = graphs if graphs is not None else generate_graphs(
            n_r, p, s, attrs=self.graph_attr_func
        )
        self.is_solved = check_solved
        self.replay_buffer = replay_buffer
        self.state: torch.Tensor = None
        self.n_step = n_step
        self.reset()

    def reset(self, g=None):
        """Resets the environment and updates the state."""
        self.state = (g or choice(self.graphs)).clone()
        self.state.step = 0
        self.state.history = []
        self.state.events_to_save = []
        return self.state

    def get_action(self, net: nn.Module, epsilon: float, device: str,
                   state=None) -> int:
        """Using the given network, decide what action to carry out using an
        epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device
            state: TODO

        Returns:
            action

        """
        state = state or self.state
        x = state.x[:, 0]
        current_solution = (x == 1).squeeze()

        if np.random.random() < epsilon:
            action = (~current_solution).float().multinomial(1)
        else:
            edge_index, node_feats = state.edge_index, state.x
            nb_batch = torch.zeros(x.size(0), dtype=torch.long)

            device = torch.device(device)
            edge_index = edge_index.to(device)
            node_feats = node_feats.to(device)
            nb_batch = nb_batch.to(device)

            q_values = net(node_feats, edge_index, nb_batch).squeeze()
            q_values[current_solution] = float("-Inf")
            _, action = torch.max(q_values, dim=0)

        return int(action.item())

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
        state=None,
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device
            state: TODO

        Returns:
            reward, done

        """
        state = state or self.state
        action = self.get_action(net, epsilon, device, state)
        if state.x[action, 0] == 1:
            return .0, False

        new_state = state.x.clone()
        new_state[action][0] = 1
        s = {i for i, x in enumerate(new_state) if x[0] == 1}
        solved = self.is_solved(state.nx, s)

        if self.graph_attr_func:
            new_state[:, 1] = self.graph_attr_func(state.edge_index, s)

        reward = -1
        reward /= len(new_state)

        exp = Experience(state.clone(), action, reward, solved, None, 0)
        state.history.append(exp)
        state.step += 1

        if state.step >= self.n_step:
            total_r = sum(s.reward for s in state.history)
            exp = state.history.pop(0)
            exp = exp._replace(new_state=Data(new_state), reward=total_r,
                               done=solved)
            del exp.state.history, exp.state.step, exp.state.events_to_save
            state.events_to_save.append(exp)

        state.x = new_state
        if solved:
            exp = state.history.pop(0)
            exp = exp._replace(new_state=Data(new_state), done=True)
            del exp.state.history, exp.state.step, exp.state.events_to_save
            state.events_to_save.append(exp)

            total_e_reward = reward * state.step
            for e in state.events_to_save:
                e = e._replace(total_reward=total_e_reward)
                self.replay_buffer.append(e)
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
        if self.state.x[action, 0] == 1:
            return .0, False

        new_state = self.state.x.clone()
        new_state[action][0] = 1
        s = {i for i, x in enumerate(new_state) if x[0] == 1}
        if self.graph_attr_func:
            new_state[:, 1] = self.graph_attr_func(self.state.edge_index, s)
        solved = self.is_solved(self.state.nx, s)
        self.state.x = new_state

        reward = -1
        reward /= len(new_state)

        return float(reward), solved


class CosineWarmupScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters, max_lr):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.start = 0
        self.max_lr = max_lr
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [self.max_lr * lr_factor for _ in self.base_lrs]

    def get_lr_factor(self, epoch):
        epoch_adj = epoch - self.start
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch_adj /
                                      (self.max_num_iters - self.start)))
        if epoch_adj <= self.warmup:
            lr_factor *= epoch_adj * 1.0 / self.warmup
        return lr_factor


class DQNLightning(LightningModule):
    def __init__(
        self,
        n: int = 10,
        p: float = .15,
        s: int = 10000,
        batch_size: int = 5000,
        delta_n: int = 10,
        lr: float = 2e-2,
        gamma: float = 0.99,
        sync_rate: int = 2**10,
        replay_size: int = 100000,
        eps_last_frame: int = 2000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        episode_length: int = 5000,
        warm_start_steps: int = 100000,
        validation_size: int = 300,
        n_step: int = 2,
        graph_attr=None,
        graphs=None,
        check_solved=None,
        tau=.001,
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

        if delta_n == n:
            delta_n += 1
        n_r = range(n, delta_n)
        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(n_r, p, s, self.buffer, n_step,
                           graph_attr_func=graph_attr, graphs=graphs,
                           check_solved=check_solved)

        model_kwargs['c_in'] = self.agent.state.x.size(dim=1)
        self.net = DQGN(**model_kwargs)
        self.target_net = DQGN(**model_kwargs)

        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)
        self.log = partial(self.log, batch_size=batch_size)
        self.s_a, self.s_b = 100, 100

    def play_until_done(self):
        state = self.agent.reset()
        for i in range(self.hparams.n):
            _, done = self.agent.play_step(self.net, epsilon=1.0, state=state)
            if done:
                break
        return i

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially
         fill up the replay buffer with experiences.

        Args:
            steps: target number of steps to collect

        """
        total_steps = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while total_steps < steps:
                # Submit a batch of tasks
                futures = [executor.submit(self.play_until_done) for _ in
                           range(8)]
                for future in concurrent.futures.as_completed(futures):
                    total_steps += future.result()
                    if total_steps >= steps:
                        break

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of
        each action as an output.

        Args:
            x: environment state
            edge_index: the incidence matrix

        Returns:
            q values

        """
        output = self.net(x, edge_index)
        return output

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the MSE loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """

        states, actions, rewards, dones, next_states = batch
        nb_batch = states.batch
        
        # Calculate the number of nodes in each graph using nb_batch
        unique_graphs, counts = nb_batch.unique(return_counts=True)
        n_per_graph = counts.tolist()

        # Reshape dynamically based on the batch graph sizes
        state_action_values = self.net(
            states.x, states.edge_index, nb_batch
        ).split(n_per_graph)
        state_action_values = torch.cat([
            values[actions[idx].long()]
            for idx, values in enumerate(state_action_values)
        ])

        with torch.no_grad():
            next_state_values = self.target_net(
                next_states.x, states.edge_index, nb_batch
            ).split(n_per_graph)
            next_state_values = torch.cat([
                values.max(0)[0] for values in next_state_values
            ])
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = (
            next_state_values * self.hparams.gamma + rewards
        )

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
        for target_param, local_param in zip(
                self.target_net.parameters(), self.net.parameters()
        ):
            # Apply the soft update formula
            target_param.data.copy_(
                self.hparams.tau * local_param.data
                + (1.0 - self.hparams.tau) * target_param.data
            )

        if self.global_step and self.global_step % self.s_a == 0:
            # state_dict = self.net.state_dict()
            # self.target_net.load_state_dict(state_dict)
            # self.s_a, self.s_b = self.s_a + self.s_b, self.s_a
            # self.log('last_sync', float(self.s_b), prog_bar=True)

            # Starting over the scheduler
            scheduler: CosineWarmupScheduler = self.lr_schedulers()
            warmup, max_iters = self.get_warmup_max_iters()
            scheduler.warmup = warmup
            scheduler.max_num_iters = max_iters
            scheduler.start = self.s_b

        self.log_dict(
            {
                "reward": reward,
                "train_loss": loss,
            }
        )
        self.log("total_reward", float(self.total_reward), prog_bar=True)
        self.log("steps", float(self.global_step), logger=False, prog_bar=True)
        last_lr = getattr(self.lr_schedulers(), '_last_lr',
                          [self.hparams.lr])[0]
        self.log("lr", last_lr, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        old_agent_state = self.agent.state
        device = self.get_device(batch)
        total_reward = 0
        val_apx_ratio = 0
        for g in batch.to_data_list():
            episode_reward = 0
            self.agent.reset(g)
            while True:
                reward, done = self.agent.play_validation_step(
                    self.net, device=device) 
                episode_reward += reward
                if done:
                    break
            total_reward += episode_reward
            sol_size = (self.agent.state.x[:, 0] == 1).sum(0)
            opt_size = (g.y == 1).sum(0)
            val_apx_ratio += sol_size / opt_size 

        self.log("val_avg_reward", total_reward/batch.num_graphs)
        self.log("val_apx_ratio", val_apx_ratio/batch.num_graphs)
        self.agent.state = old_agent_state

    def get_warmup_max_iters(self):
        return .05 * self.s_a, self.s_a

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        warmup, max_iters = self.get_warmup_max_iters()
        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer,
                                             warmup=warmup,
                                             max_iters=max_iters,
                                             max_lr=self.hparams.lr)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

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

