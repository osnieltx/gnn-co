from random import choice
from typing import Iterable, Callable
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.distributions import Categorical
from torch.optim.optimizer import Optimizer
from torch.utils.data import IterableDataset

from dql import DQGN
from graph import generate_graphs, is_ds, dominating_potential
from pyg import geom_data


class Agent:
    def __init__(
        self, n_r: range, p: float, s: int, actor, critic, graph_attr: List,
            graphs=None, device='cuda'
    ) -> None:
        """Base Agent class handling the interaction with the environment.

        """
        graphs = graphs or generate_graphs(n_r, p, s, attrs=graph_attr)
        self.graphs = [g.to(device) for g in graphs]
        self.state: geom_data.Data = None
        self.reset()
        self.actor = actor
        self.critic = critic

    def reset(self, g=None) -> None:
        """Resets the environment and updates the state."""
        self.state = g or choice(self.graphs).clone()
        self.state.step = 0

    @staticmethod
    def _prepare_forward(device: str, nb_batch, state):
        edge_index, node_feats = state.edge_index, state.x
        if nb_batch is None:
            nb_batch = torch.zeros(node_feats.size(0), dtype=torch.long)

        device = torch.device(device)
        edge_index = edge_index.to(device)
        node_feats = node_feats.to(device)
        nb_batch = nb_batch.to(device)
        return edge_index, node_feats, nb_batch

    def get_action(self, device: str = 'cpu', state: geom_data.Data = None,
                   nb_batch: torch.Tensor = None) -> \
            Tuple[Categorical, torch.Tensor]:
        state = state or self.state
        x = state.x[:, 0]
        current_solution = (x == 1).squeeze()
        edge_index, node_feats, nb_batch = self._prepare_forward(device,
                                                                 nb_batch,
                                                                 state)

        logits = self.actor(node_feats, edge_index, nb_batch).squeeze()
        logits = logits.clone()  # keep the intermediate tensor for autograd
        logits[current_solution] = float("-Inf")
        # Find number of graphs and max nodes per graph
        num_graphs = nb_batch.max().item() + 1
        max_nodes = max((nb_batch == i).sum().item() for i in range(num_graphs))
        # Create a padded tensor with -Inf
        batch_outputs = torch.full((num_graphs, max_nodes), float('-inf'),
                                   device=device)
        # Fill the tensor with actual values
        for i in range(num_graphs):
            graph_nodes = logits[nb_batch == i]  # Nodes belonging to graph i
            batch_outputs[i, :len(graph_nodes)] = graph_nodes
        pi = Categorical(logits=batch_outputs)
        value = pi.sample()

        return pi, value

    def get_value(self, device: str = 'cpu', state: geom_data.Data = None,
                  nb_batch: torch.Tensor = None) -> float:
        state = state or self.state
        edge_index, node_feats, nb_batch = self._prepare_forward(device,
                                                                 nb_batch,
                                                                 state)

        # TODO check output
        value = self.critic(node_feats, edge_index, nb_batch).squeeze()

        return value

    @torch.no_grad()
    def play_step(self, device: str = "cpu", reset_when_solved=True, state=None) -> Tuple:
        """Carries out a single interaction step between the agent and the
        environment.

        Args:
            device: current device

        Returns:
            reward, done
            :param device:
            :param reset_when_solved:

        """
        state = state or self.state

        pi, action = self.get_action(device, state)
        log_prob = pi.log_prob(action)

        value = self.get_value(device, state)

        if state.x[action, 0] == 1:
            return .0, False

        reward = -1/state.x.size(0)
        new_state = state.x.clone()
        new_state[action, 0] = 1
        s = {i for i, x in enumerate(new_state) if x[0] == 1}
        solved = is_ds(state.nx, s)
        new_state[:, 1] = dominating_potential(state.edge_index, s)
        state.step += 1
        state.x = new_state
        if reset_when_solved and solved:
            self.reset()

        return action, log_prob, value, new_state, float(reward), solved


class ExperienceSourceDataset(IterableDataset):
    """
    Implementation from PyTorch Lightning Bolts:
    https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/datamodules/experience_source.py

    Basic experience source dataset. Takes a generate_batch function that returns an iterator.
    The logic for the experience source and how the batch is generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable):
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterable:
        iterator = self.generate_batch()
        return iterator


class PPO(pl.LightningModule):
    """
    PyTorch Lightning implementation of `PPO
    <https://arxiv.org/abs/1707.06347>`_
    Paper authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov

    Train:
        trainer = Trainer()
        trainer.fit(model)
    Note:
        This example is based on:
        https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py
        https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/rl/reinforce_model.py

    """
    def __init__(
        self,
        n: int = 10,
        p: float = .15,
        s: int = 10000,
        delta_n: int = 10,
        graph_attr: List = [],
        gamma: float = 0.99,
        lam: float = 0.95,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        max_episode_len: float = 1000,
        batch_size: int = 512,
        steps_per_epoch: int = 2048,
        nb_optim_iters: int = 4,
        clip_ratio: float = 0.2,
        device='cuda',
        **model_kwargs
    ) -> None:

        """

        Args:
            gamma: discount factor lam: advantage discount factor (lambda
                in the paper)
            lr_actor: learning rate of actor network
            lr_critic: learning rate of critic network
            max_episode_len: maximum number interactions (actions) in an episode
            batch_size:  batch_size when training network - can simulate number
                of policy updates performed per epoch
            steps_per_epoch: how many action-state pairs to roll out for
                trajectory collection per epoch
            nb_optim_iters: how many steps of gradient descent to perform on
                each batch
            clip_ratio: hyperparameter for clipping in the policy objective

        """
        super().__init__()

        self.automatic_optimization = False
        # Hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.steps_per_epoch = steps_per_epoch
        self.nb_optim_iters = nb_optim_iters
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.max_episode_len = max_episode_len
        self.clip_ratio = clip_ratio
        self.save_hyperparameters()

        model_kwargs['c_in'] = 1 + len(graph_attr)
        # value network
        self.critic = DQGN(**model_kwargs, aggr_out_by_graph=True)
        # policy network (agent)
        self.actor = DQGN(**model_kwargs)

        if delta_n == n:
            delta_n += 1
        n_r = range(n, delta_n)
        self.agent = Agent(n_r, p, s, self.actor, self.critic,
                           graph_attr=graph_attr, device=device)

        self.batch_states = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []

        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []

        self.episode_step = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_reward = 0

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Passes in a state x through the network and returns the policy and a
        sampled action Args: x: environment state Returns: Tuple of policy
        and action
        """
        nb_batch = torch.zeros(x.size(0), dtype=torch.long)
        logits = self.actor(x, edge_index, nb_batch)
        pi = Categorical(logits=logits)
        action = pi.sample()

        value = self.critic(x, edge_index, nb_batch)

        return pi, action, value

    def discount_rewards(self, rewards: List[float], discount: float)\
            -> List[float]:
        """Calculate the discounted rewards of all rewards in list
        Args:
            rewards: list of rewards/advantages
        Returns:
            list of discounted rewards/advantages
            :param discount:
        """
        assert isinstance(rewards[0], float)

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * discount) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def calc_advantage(self, rewards: List[float], values: List[float],
                       last_value: float) -> List[float]:
        """Calculate the advantage given rewards, state values, and the last
        value of episode
        Args:
            rewards: list of episode rewards
            values: list of state values from critic
            last_value: value of last state of episode
        Returns: list of advantages
        """
        rews = rewards + [last_value]
        vals = values + [last_value]
        # GAE
        delta = [rews[i] + self.gamma * vals[i + 1] - vals[i]
                 for i in range(len(rews) - 1)]
        adv = self.discount_rewards(delta, self.gamma * self.lam)

        return adv

    def train_batch(
            self,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Contains the logic for generating trajectory data to train policy and
        value network
        Yield:
           Tuple of Lists containing tensors for states, actions, log probs,
           qvals and advantage
        """

        for step in range(self.steps_per_epoch):

            init_state = self.agent.state.clone()
            result = self.agent.play_step(self.device)
            action, log_prob, value, next_state, reward, done = result

            self.episode_step += 1

            self.batch_states.append(init_state)
            self.batch_actions.append(action)
            self.batch_logp.append(log_prob)

            self.ep_rewards.append(reward)
            self.ep_values.append(value.item())

            epoch_end = step == (self.steps_per_epoch - 1)
            terminal = len(self.ep_rewards) == self.max_episode_len

            if epoch_end or done or terminal:
                # if trajectory ends abtruptly, boostrap value of next state
                if (terminal or epoch_end) and not done:
                    # not sure if this is actually necessary
                    with torch.no_grad():
                        value = self.agent.get_value(self.device,
                                                     self.agent.state)
                        last_value = value.item()
                        steps_before_cutoff = self.episode_step
                else:
                    last_value = 0
                    steps_before_cutoff = 0

                # discounted cumulative reward
                self.batch_qvals += self.discount_rewards(
                    self.ep_rewards + [last_value], self.gamma)[:-1]
                # advantage
                self.batch_adv += self.calc_advantage(
                    self.ep_rewards, self.ep_values, last_value)
                # logs
                self.epoch_rewards.append(sum(self.ep_rewards))
                # reset params
                self.ep_rewards = []
                self.ep_values = []
                self.episode_step = 0

            if epoch_end:
                train_data = zip(
                    self.batch_states, self.batch_actions, self.batch_logp,
                    self.batch_qvals, self.batch_adv)

                for state, action, logp_old, qval, adv in train_data:
                    yield state, action, logp_old, qval, adv

                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_adv.clear()
                self.batch_logp.clear()
                self.batch_qvals.clear()

                # logging
                self.avg_reward = sum(self.epoch_rewards) / self.steps_per_epoch

                # if epoch ended abruptly, exlude last cut-short episode to
                # prevent stats skewness
                epoch_rewards = self.epoch_rewards
                if not done:
                    epoch_rewards = epoch_rewards[:-1]

                total_epoch_reward = sum(epoch_rewards)
                nb_episodes = len(epoch_rewards)

                self.avg_ep_reward = total_epoch_reward / nb_episodes
                self.avg_ep_len = ((self.steps_per_epoch - steps_before_cutoff)
                                   / nb_episodes)

                self.epoch_rewards.clear()

    def actor_loss(self, state, action, logp_old, qval, adv) -> torch.Tensor:
        pi, _ = self.agent.get_action(state=state, device=self.device,
                                      nb_batch=state.batch)
        logp = pi.log_prob(action.squeeze())
        ratio = torch.exp(logp - logp_old.squeeze())
        clip_adv = (torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    * adv)
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_actor

    def critic_loss(self, state, action, logp_old, qval, adv) -> torch.Tensor:
        value = self.agent.get_value(state=state, device=self.device,
                                     nb_batch=state.batch)
        loss_critic = (qval - value).pow(2).mean()
        return loss_critic

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx
    ):
        """
        Carries out a single update to actor and critic network from a batch
        of replay buffer.

        Args:
            batch: batch of replay buffer/trajectory data
            batch_idx: not used
        Returns:
            loss
        """
        states, action, old_logp, qval, adv = batch
        nb_batch = states.batch

        # Calculate the number of nodes in each graph using nb_batch
        unique_graphs, counts = nb_batch.unique(return_counts=True)
        n_per_graph = counts.tolist()

        # normalize advantages
        adv = (adv - adv.mean())/adv.std()

        self.log("avg_ep_len", self.avg_ep_len, prog_bar=True, on_step=False,
                 on_epoch=True)
        self.log("avg_ep_reward", self.avg_ep_reward, prog_bar=True,
                 on_step=False, on_epoch=True)
        self.log("avg_reward", self.avg_reward, prog_bar=True, on_step=False,
                 on_epoch=True)

        actor_opt, critic_opt = self.optimizers()

        loss_actor = self.actor_loss(states, action, old_logp, qval, adv)
        self.log('loss_actor', loss_actor, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        actor_opt.zero_grad()
        self.manual_backward(loss_actor)
        actor_opt.step()

        loss_critic = self.critic_loss(states, action, old_logp, qval, adv)
        self.log('loss_critic', loss_critic, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)
        critic_opt.zero_grad()
        self.manual_backward(loss_critic)
        critic_opt.step()

    def validation_step(self, batch, batch_idx):
        old_agent_state = self.agent.state
        total_reward = 0
        val_apx_ratio = 0
        for g in batch.to_data_list():
            episode_reward = 0
            self.agent.reset(g)
            while True:
                reward, done = self.agent.play_step(
                    device=self.device, reset_when_solved=False)[-2:]
                episode_reward += reward
                if done:
                    break
                if (episode_reward < reward * self.hparams.delta_n):
                    raise ValueError('Extreme episode reward')
            total_reward += episode_reward
            sol_size = (self.agent.state.x[:,0] == 1).sum(0)
            opt_size = (g.y == 1).sum(0)
            val_apx_ratio += sol_size / opt_size 

        self.log("val_avg_reward", total_reward/batch.num_graphs)
        self.log("val_apx_ratio", val_apx_ratio/batch.num_graphs)
        self.agent.state = old_agent_state

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        optimizer_critic = optim.Adam(self.critic.parameters(),
                                      lr=self.lr_critic)

        return optimizer_actor, optimizer_critic

    def optimizer_step(self, *args, **kwargs):
        """
        Run 'nb_optim_iters' number of iterations of gradient descent on
        actor and critic for each data sample.
        """
        for i in range(self.nb_optim_iters):
            super().optimizer_step(*args, **kwargs)

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving
        experiences """
        dataset = ExperienceSourceDataset(self.train_batch)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader()
