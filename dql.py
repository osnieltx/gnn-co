import threading
from collections import OrderedDict, deque, namedtuple
from functools import partial
from random import choice
from typing import Iterator, List, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.optim import Adam, Optimizer, lr_scheduler
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import degree
from torch.utils.data.dataset import IterableDataset

import s2v
from graph import generate_graphs
from pyg import geom_nn


gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv,
    "s2v": s2v.Structure2VecConv
}


class DQGN(nn.Module):
    def __init__(self, c_in, c_hidden=64, c_out=1,
                 num_layers=5, layer_name="s2v", dp_rate=None,
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


class DQGNS2V(nn.Module):
    def __init__(self, c_in, c_hidden=64, c_out=1,
                 num_iterations=5, dp_rate=None,
                 aggr_out_by_graph=False):
        """
        Updated for Structure2Vec.
        Note: 'num_layers' is renamed to 'num_iterations' for clarity,
        as we are repeating the same layer, not stacking distinct ones.
        """
        super().__init__()

        self.aggr_out_by_graph = aggr_out_by_graph
        self.num_iterations = num_iterations
        self.c_hidden = c_hidden

        # --- THE S2V UPDATE ---
        # Instantiate exactly ONE convolutional layer. Weights are shared.
        self.conv = s2v.Structure2VecConv(in_channels=c_in,
                                          out_channels=c_hidden)

        # Optional dropout
        self.dropout = nn.Dropout(dp_rate) if dp_rate else None

        # --- YOUR ORIGINAL TRANSFORMS ---
        # Note: The input to this is now c_hidden (the size of mu)
        self.node_transform = nn.Linear(c_hidden, c_hidden, bias=False)

        if self.aggr_out_by_graph:
            self.grph_transform = nn.Linear(c_hidden, c_out, bias=False)
        else:
            self.grph_transform = nn.Linear(c_hidden, c_hidden, bias=False)
            self.aggr_transform = nn.Linear(2 * c_hidden, c_out, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge_index, nb_batch):
        """
        Inputs:
        x - Input features per node
        edge_index - Graph connectivity
        nb_batch - Batch indices for nodes
        """

        # --- THE S2V FORWARD PASS ---
        # 1. Initialize the hidden state (mu) to zeros
        mu = torch.zeros((x.size(0), self.c_hidden), device=x.device)

        # 2. S2V Recursive Updates with non-linearity
        for _ in range(self.num_iterations):
            mu = self.conv(x, edge_index, mu)

        if self.dropout is not None:
            mu = self.dropout(mu)

        # 3. Decoupled Global and Local Transformations
        # Pool the RAW mu for global context
        graph_pool = global_add_pool(mu, nb_batch)

        # Local part (theta_7 * mu_v)
        local_part = self.node_transform(mu)
        # Global part (theta_6 * sum(mu_u))
        # index_select instead of fancy indexing: the latter has no MPS kernel.
        global_part = torch.index_select(self.grph_transform(graph_pool), 0, nb_batch)

        # 4. Concatenate and apply final ReLU/Linear layer
        out = torch.cat((local_part, global_part), dim=1)
        out = self.relu(out)
        out = self.aggr_transform(out)  # Final theta_5 projection

        return out

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
        """
        Samples experiences uniformly from the buffer to align with standard
        experience replay.
        """
        # 1. Randomly select indices without replacement
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        # 2. Gather samples based on indices
        # We maintain the order: state, action, reward, done, next_state
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

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 1) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield (
                states[i],
                torch.as_tensor(actions[i], dtype=torch.long),
                torch.as_tensor(rewards[i], dtype=torch.float),
                torch.as_tensor(dones[i], dtype=torch.bool),
                new_states[i]
            )


class Agent:
    def __init__(
        self, n_r: range, p: float, s: int, replay_buffer: ReplayBuffer,
        n_step: int, graphs=None, graph_attr_func=None, check_solved=None, max_n=100
    ) -> None:
        self.p = p
        self.s = s
        self.graph_attr_func = graph_attr_func
        self.is_solved = check_solved
        self.replay_buffer = replay_buffer
        self.n_step = n_step
        self.max_n = max_n
        self.graphs = graphs if graphs is not None else generate_graphs(
            n_r, p, s, attrs=self.graph_attr_func
        )
        self.state: torch.Tensor = None
        self.reset()

    def update_graphs(self, n_r: range, graphs=None, cumulative: bool = False) -> None:
        """Regenerates or appends to the graph distribution for the new curriculum stage."""
        new_graphs = graphs if graphs is not None else generate_graphs(
            n_r, self.p, self.s, attrs=self.graph_attr_func
        )
        if cumulative:
            self.graphs.extend(new_graphs)
        else:
            self.graphs = new_graphs
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
        """Carries out a single interaction step between the agent and the
        environment.

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

        # 1. Update State
        new_node_feats = state.x.clone()
        new_node_feats[action][0] = 1
        selected_mask = new_node_feats[:, 0] == 1
        solved = self.is_solved(state.edge_index, selected_mask)

        if self.graph_attr_func:
            s = {i for i, x in enumerate(new_node_feats) if x[0] == 1}
            new_node_feats[:, 1] = self.graph_attr_func(state.edge_index, s)

        reward = -1 / self.max_n

        # 2. Append to current history
        clean_state = Data(x=state.x.clone(),
                           edge_index=state.edge_index.clone())
        exp = Experience(clean_state, action, reward, solved, None, 0)
        state.history.append(exp)
        state.step += 1

        # 3. Standard n-step: if we have enough history, pop the oldest
        if len(state.history) >= self.n_step:
            total_r = sum(s.reward for s in state.history)
            old_exp = state.history.pop(0)
            clean_new_state = Data(x=new_node_feats.clone(),
                                   edge_index=state.edge_index.clone())
            new_exp = old_exp._replace(new_state=clean_new_state,
                                       reward=total_r, done=solved)
            state.events_to_save.append(new_exp)

        # 4. Handle Termination
        state.x = new_node_feats
        if solved:
            while state.history:
                total_r = sum(s.reward for s in state.history)
                old_exp = state.history.pop(0)
                clean_new_state = Data(x=new_node_feats.clone(),
                                       edge_index=state.edge_index.clone())
                new_exp = old_exp._replace(new_state=clean_new_state,
                                           reward=total_r, done=True)
                state.events_to_save.append(new_exp)

            # Record the final solution size
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
        """Carries out a single interaction step between the agent and the
        environment.

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
        if self.graph_attr_func:
            s = {i for i, x in enumerate(new_state) if x[0] == 1}
            new_state[:, 1] = self.graph_attr_func(self.state.edge_index, s)
        selected_mask = new_state[:, 0] == 1
        solved = self.is_solved(self.state.edge_index, selected_mask)
        self.state.x = new_state

        reward = -1 / self.max_n

        return float(reward), solved


class CosineWarmupScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, stage_max_iters, max_lr):
        self.warmup = warmup
        self.stage_max_iters = stage_max_iters
        self.stage_start = 0
        self.stage_idx = 0
        self.max_lr = max_lr
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [self.max_lr * lr_factor for _ in self.base_lrs]

    def get_lr_factor(self, epoch):
        epoch_adj = epoch - self.stage_start

        # Prevent negative values if stage exceeds max_iters
        epoch_adj = min(epoch_adj, self.stage_max_iters)

        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch_adj / self.stage_max_iters))

        # Linear warmup phase
        if epoch_adj <= self.warmup:
            lr_factor *= epoch_adj * 1.0 / self.warmup

        # Halve the maximum learning rate for curriculum stages > 0
        if self.stage_idx > 0:
            lr_factor *= 0.5

        return lr_factor

    def advance_stage(self, current_epoch):
        """Called by the LightningModule when curriculum advances."""
        self.stage_start = current_epoch
        self.stage_idx += 1


class DQNLightning(LightningModule):
    def __init__(
            self,
            n_sizes: List[Union[int, range, tuple]] = [10, 20, 30, 40, 50, 60],
            curriculum_mode: str = "replace",  # "replace" or "cumulative"
            target_apx_ratio=1.005,
            stage_warm_start_steps: int = 1000,
            p: float = 0.15,
            s: int = 10000,
            batch_size: int = 128,
            lr: float = 7e-4,
            gamma: float = 1.0,
            sync_rate: int = 1000,
            replay_size: int = 100000,
            eps_last_frame: int = 15000,
            eps_start: float = 1.0,
            eps_end: float = 0.05,
            warm_start_steps: int = 2000,
            n_step: int = 5,
            graph_attr=None,
            check_solved=None,
            max_epochs: int = 2500,
            **model_kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Parse curriculum configuration
        self.curriculum_stages = [
            r if isinstance(r, range)
            else range(r[0], r[1]) if isinstance(r, (tuple, list))
            else range(r, r + 1)
            for r in n_sizes
        ]
        max_n = self.curriculum_stages[0].stop -1
        print(f'{max_n=}')
        self.current_stage_idx = 0
        self.stage_start_step = 0

        # Initialize Replay Buffer and Agent with first curriculum stage
        initial_n_range = self.curriculum_stages[0]
        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(
            n_r=initial_n_range,
            p=p,
            s=s,
            replay_buffer=self.buffer,
            n_step=n_step,
            graph_attr_func=graph_attr,
            check_solved=check_solved,
            max_n=max_n,
        )

        model_kwargs['c_in'] = self.agent.state.x.size(dim=1)
        self.net = DQGNS2V(**model_kwargs)
        self.target_net = DQGNS2V(**model_kwargs)

        self.total_reward = 0
        self.episode_reward = 0

        # Initial buffer population
        self.populate(self.hparams.warm_start_steps)
        self.log = partial(self.log, batch_size=batch_size)

    def advance_curriculum(self) -> None:
        """Transitions the agent to the next graph size stage."""
        if self.current_stage_idx >= len(self.curriculum_stages) - 1:
            return

        self.current_stage_idx += 1
        new_range = self.curriculum_stages[self.current_stage_idx]

        # self.agent.max_n = new_range.stop - 1

        # 1. Update Agent graph distribution (extend or replace)
        is_cumulative = (self.hparams.curriculum_mode == "cumulative")
        self.agent.update_graphs(new_range, cumulative=is_cumulative)

        # 2. Warm-start the replay buffer
        # if self.hparams.stage_warm_start_steps > 0:
        #     self.populate(self.hparams.stage_warm_start_steps)

        # 3. Synchronize target network on distribution shift
        self.target_net.load_state_dict(self.net.state_dict())

        # 4. Reset Epsilon tracker
        # self.stage_start_step = self.global_step

        # 5. Reset LR Scheduler
        # scheduler = self.lr_schedulers()
        # if scheduler is not None:
        #     actual_scheduler = scheduler.scheduler if hasattr(scheduler, 'scheduler') else scheduler
        #     actual_scheduler.advance_stage(self.current_epoch)


    def on_validation_epoch_end(self) -> None:
        stage_apx = self.trainer.callback_metrics.get("val_apx_ratio/stage")
        if stage_apx is not None and stage_apx <= self.hparams.target_apx_ratio:
            self.advance_curriculum()
        # Log current curriculum metadata
        current_range = self.curriculum_stages[self.current_stage_idx]
        self.log("curriculum/stage", float(self.current_stage_idx), prog_bar=True)
        self.log("curriculum/graph_size_min", float(current_range.start), prog_bar=True)

    def populate(self, steps: int = 1000) -> None:
        """
        Avoids cloning by using raw tensors for logic, only creating Data
        objects for the buffer.
        Uses Agent.is_solved to find solve-steps for an entire batch.
        """
        total_added = 0
        internal_batch_size = 64
        loader = DataLoader(self.agent.graphs, batch_size=internal_batch_size,
                            shuffle=True)

        for batch in loader:
            if total_added >= steps:
                break

            batch = batch.to(self.device)
            num_graphs = batch.num_graphs
            num_nodes = batch.x.size(0)

            # 1. Generate random priorities for every node in the batch
            priorities = torch.rand(num_nodes, device=self.device)

            # 2. Vectorized Truncation: Find when each graph becomes a solution
            solved_at_step = torch.full((num_graphs,), -1, dtype=torch.long,
                                        device=self.device)

            # We iterate through possible set sizes
            max_possible_steps = degree(batch.batch, num_graphs, dtype=torch.long).max().item()
            for s in range(1, max_possible_steps + 1):
                # Mask nodes that are in the top 's' priorities within their
                # respective graph. This is a vectorized way to simulate picking
                # nodes randomly one-by-one.
                current_mask = self._get_top_k_mask(batch.batch, priorities, s)

                is_solved = self.agent.is_solved(batch.edge_index, current_mask,
                                                 batch.batch, num_graphs)

                # Record completion for graphs that just hit 'solved' status
                just_finished = is_solved & (solved_at_step == -1)
                solved_at_step[just_finished] = s

                if is_solved.all():
                    break

            # 3. Assemble Experiences
            total_added += self._push_batch_to_buffer(batch, priorities,
                                                      solved_at_step)

    def _get_top_k_mask(self, batch_idx, priorities, k):
        """Vectorized helper to select top k priority nodes per graph."""
        # This creates a mask where True = node is selected at this step
        # Sort priorities within each graph group
        ranks = []
        for g_i in range(batch_idx.max() + 1):
            g_priorities = priorities[batch_idx == g_i]
            # Handle cases where graph has fewer nodes than k
            actual_k = min(k, len(g_priorities))
            _, top_indices = torch.topk(g_priorities, actual_k)

            mask = torch.zeros(len(g_priorities), dtype=torch.bool,
                               device=self.device)
            mask[top_indices] = True
            ranks.append(mask)
        return torch.cat(ranks)

    def _push_batch_to_buffer(self, batch, priorities, solved_at_step):
        """
        Formats the raw rollout into Experience tuples
        """
        added_count = 0
        for g_i in range(batch.num_graphs):
            # Extract nodes and priorities for this specific graph instance
            g_mask = (batch.batch == g_i)
            g_nodes = torch.where(g_mask)[0]
            g_priorities = priorities[g_mask]

            # Sort actions by priority (random greedy simulation)
            actions = g_nodes[torch.argsort(g_priorities, descending=True)]
            solve_limit = solved_at_step[g_i]

            # Only process graphs that were actually solved
            if solve_limit <= 0:
                continue

            final_actions = actions[:solve_limit]

            for i in range(len(final_actions)):
                n = self.hparams.n_step
                reward_per_step = -1.0 / self.agent.max_n

                actual_n = min(n, len(final_actions) - i)
                total_n_reward = reward_per_step * actual_n

                # We use local indices (0 to N-1) relative to the graph's start
                local_current_indices = (final_actions[:i] - g_nodes[0]).long()
                local_next_indices = (
                            final_actions[:i + actual_n] - g_nodes[0]).long()

                # Initialize feature vectors
                state_x = torch.zeros((len(g_nodes), 1), device=self.device)
                state_x[local_current_indices, 0] = 1

                next_state_x = state_x.clone()
                next_state_x[local_next_indices, 0] = 1

                # Get local edge structure for this graph [cite: 1, 109-112]
                local_edges = self._get_local_edges(batch, g_i)

                exp = Experience(
                    state=Data(x=state_x.cpu(), edge_index=local_edges.cpu()),
                    action=(final_actions[i] - g_nodes[0]).item(),
                    # Store as local index
                    reward=total_n_reward,
                    done=(i + actual_n >= len(final_actions)),
                    new_state=Data(x=next_state_x.cpu(),
                                   edge_index=local_edges.cpu()),
                    total_reward=float(len(final_actions) * reward_per_step)
                )
                self.buffer.append(exp)  #
                added_count += 1

        return added_count

    def _get_local_edges(self, batch, g_i):
        """Slices the batch edge_index to get edges belonging to graph g_i."""
        edge_mask = (batch.batch[batch.edge_index[0]] == g_i)
        local_edges = batch.edge_index[:, edge_mask]
        # Normalize edges to start from 0 for the local graph context
        offset = torch.where(batch.batch == g_i)[0][0]
        return local_edges - offset

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
        """Calculates the DDQN MSE loss using a mini batch from the replay buffer."""
        states, actions, rewards, dones, next_states = batch
        nb_batch = states.batch

        # Calculate the number of nodes in each graph using nb_batch
        unique_graphs, counts = nb_batch.unique(return_counts=True)
        n_per_graph = counts.tolist()

        # Current state Q-values
        state_action_values = self.net(
            states.x, states.edge_index, nb_batch
        ).split(n_per_graph)

        state_action_values = torch.cat([
            values[actions[idx].long()]
            for idx, values in enumerate(state_action_values)
        ])

        with torch.no_grad():
            # 1. Use ONLINE network to select the best next actions
            online_next_values = self.net(
                next_states.x, states.edge_index, nb_batch
            ).split(n_per_graph)

            # 2. Use TARGET network exclusively to evaluate those selected actions
            target_next_values = self.target_net(
                next_states.x, states.edge_index, nb_batch
            ).split(n_per_graph)

            # We need the node features split so we know which nodes are already selected
            next_state_feats = next_states.x.split(n_per_graph)

            masked_next_values = []
            for idx in range(len(n_per_graph)):
                # Mask illegal actions in the online network
                is_selected = next_state_feats[idx][:, 0] == 1

                valid_online_values = online_next_values[idx].clone()
                valid_online_values[is_selected] = float("-Inf")

                # argmax: Find the index of the best valid action according to the online network
                best_action_idx = valid_online_values.argmax(0)

                # Evaluate that specific action's value using the target network
                eval_val = target_next_values[idx][best_action_idx]
                masked_next_values.append(eval_val)

            next_state_values = torch.cat(masked_next_values).squeeze(-1)
            next_state_values[dones] = 0.0

        expected_state_action_values = (
                next_state_values * self.hparams.gamma + rewards
        )

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def get_epsilon(self) -> float:
        steps_in_stage = self.global_step - self.stage_start_step

        # Start at 1.0 for stage 0, and 0.5 for all subsequent stages
        current_start = self.hparams.eps_start if self.current_stage_idx == 0 else (self.hparams.eps_start * 0.5)

        if steps_in_stage > self.hparams.eps_last_frame:
            return self.hparams.eps_end

        return current_start - (steps_in_stage / self.hparams.eps_last_frame) * (current_start - self.hparams.eps_end)

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
        epsilon = self.get_epsilon()
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

        # # Soft update of target network
        # for target_param, local_param in zip(
        #         self.target_net.parameters(), self.net.parameters()
        # ):
        #     # Apply the soft update formula
        #     target_param.data.copy_(
        #         self.hparams.tau * local_param.data
        #         + (1.0 - self.hparams.tau) * target_param.data
        #     )

        # periodic hard sync
        if self.global_step and self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        # if self.global_step and self.global_step % self.s_a == 0:
        #     state_dict = self.net.state_dict()
        #     self.target_net.load_state_dict(state_dict)
        #     self.s_a, self.s_b = self.s_a + self.s_b, self.s_a
        #     self.log('last_sync', float(self.s_b), prog_bar=True)

            # Starting over the scheduler
            # scheduler: CosineWarmupScheduler = self.lr_schedulers()
            # warmup, max_iters = self.get_warmup_max_iters()
            # scheduler.warmup = warmup
            # scheduler.max_num_iters = max_iters
            # scheduler.start = self.s_b

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
        device = self.get_device(batch)
        batch = batch.to(device)
        num_graphs = batch.num_graphs

        # Initialize batch tracking and FORCE a clean state
        current_x = batch.x.clone().to(self.device)
        current_x[:, 0] = 0  # Wipes any leaked solutions from the dataloader

        unsolved_mask = torch.ones(num_graphs, dtype=torch.bool, device=device)
        total_steps = torch.zeros(num_graphs, device=device)

        # Vectorized (MPS-friendly) per-graph argmax selection.
        # torch.bincount/unique/sort/index_put_/masked-fancy-indexing have no
        # MPS kernel in this torch version, so this avoids all of them in
        # favor of gather/logical_*/masked_fill/dense max(dim=...), which do
        # run natively on MPS.
        graph_ids = torch.arange(num_graphs, device=device)
        neg_inf = torch.tensor(float("-inf"), device=device)
        # (num_nodes, num_graphs) membership mask, built once per validation batch.
        node_graph_mask = batch.batch.unsqueeze(1) == graph_ids.unsqueeze(0)

        # Parallel rollout: Loop until every graph in the batch is solved
        while unsolved_mask.any():
            # 1. Batch Forward Pass
            q_values = self.net(current_x, batch.edge_index, batch.batch).squeeze()

            # 2. Ignore selected nodes and nodes in already solved graphs
            is_selected = current_x[:, 0] == 1
            node_unsolved = torch.gather(unsolved_mask, 0, batch.batch)
            nodes_in_solved_graphs = torch.logical_not(node_unsolved)
            q_values = q_values.masked_fill(
                torch.logical_or(is_selected, nodes_in_solved_graphs), float("-inf"))

            # 3. Greedy selection per graph (dense max-reduction, no Python loop)
            masked_q = torch.where(node_graph_mask, q_values.unsqueeze(1), neg_inf)
            seg_max, _ = masked_q.max(dim=0)
            node_seg_max = torch.gather(seg_max, 0, batch.batch)
            is_chosen = torch.logical_and(q_values == node_seg_max, node_unsolved)

            current_x[:, 0] = torch.logical_or(is_selected, is_chosen).to(current_x.dtype)
            total_steps += unsolved_mask.to(total_steps.dtype)

            # 4. Batch-wide Vectorized Check
            solved_mask = self.agent.is_solved(
                batch.edge_index,
                current_x[:, 0] == 1,
                batch_idx=batch.batch,
                num_graphs=num_graphs
            )
            unsolved_mask = torch.logical_not(solved_mask)

        # 5. Final Metrics
        sol_sizes = total_steps
        opt_sizes = global_add_pool((batch.y == 1).float(), batch.batch).squeeze()

        val_apx_ratio = sol_sizes / opt_sizes
        val_avg_reward = -sol_sizes.sum() / num_graphs

        self.log("val_avg_reward", val_avg_reward.mean())
        self.log("val_apx_ratio_all", val_apx_ratio.mean())

        # Calculate the number of nodes in each graph. This tiny (num_graphs,)
        # breakdown loop runs once per validation epoch, so it's cheap enough
        # to just move to CPU rather than lean on MPS's fallback for
        # unique()/boolean-mask indexing (neither has an MPS kernel).
        graph_sizes = degree(batch.batch, num_graphs, dtype=torch.long).cpu()
        val_apx_ratio_cpu = val_apx_ratio.detach().cpu()

        # Log apx-ratio for EVERY individual graph size in the validation set
        for size_val in graph_sizes.unique().tolist():
            size_mask = graph_sizes == size_val
            self.log(f"val_apx_ratio/{size_val}", val_apx_ratio_cpu[size_mask].mean())

        # Isolate the metric for the current curriculum stage to trigger progression.
        # Stays on CPU alongside graph_sizes/val_apx_ratio_cpu (see above).
        current_range = self.curriculum_stages[self.current_stage_idx]
        stage_mask = torch.tensor([size.item() in current_range for size in graph_sizes],
                                  dtype=torch.bool)

        if stage_mask.any():
            self.log("val_apx_ratio/stage", val_apx_ratio_cpu[stage_mask].mean())

        # Log cumulative ratio across all stages unlocked so far
        if self.hparams.curriculum_mode == "cumulative":
            active_ranges = self.curriculum_stages[: self.current_stage_idx + 1]
            cum_mask = torch.tensor(
                [any(size.item() in r for r in active_ranges) for size in graph_sizes],
                dtype=torch.bool,
            )
            if cum_mask.any():
                self.log("val_apx_ratio/cumulative", val_apx_ratio_cpu[cum_mask].mean())

    def get_warmup_max_iters(self):
        return .05 * self.s_a, self.s_a

    # def configure_optimizers(self) -> dict:
    #     optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
    #     cos_warmup_scheduler = CosineWarmupScheduler(
    #         optimizer=optimizer,
    #         warmup=.05 * self.hparams.max_epochs,
    #         stage_max_iters=self.hparams.max_epochs,
    #         max_lr=self.hparams.lr
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": cos_warmup_scheduler,
    #             "interval": "step"  # Ensure it updates per training step
    #         }
    #     }

    def configure_optimizers(self) -> dict:
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)

        # Matches the S2V-DQN 0.95 exponential decay factor
        scheduler = StepLR(optimizer, step_size=5000, gamma=0.95)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"  # Applies the step schedule against global_step
            }
        }

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving
        experiences."""
        dataset = RLDataset(self.buffer, sample_size=self.hparams.batch_size)
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
        """Retrieve device currently being used by the module (e.g. cpu/cuda/mps)."""
        return self.device

