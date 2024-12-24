import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax
from torch.distributions import Categorical
import bisect
import random
from torch_geometric.nn import GCNConv
import torch_geometric

from models.gcn import *
from models.fully_connected import *
from models.fully_connected_graph import *

class PiApprox(object):

    def __init__(self, state_dimensions, num_actions, learning_rate, network):

        self._state_dimensions = state_dimensions
        self._num_actions = num_actions
        self._learning_rate = learning_rate
        self._network = network(state_dimensions, num_actions)
        self._optimizer = torch.optim.Adam(self._network.parameters(), learning_rate, [0.9, 0.999])
        self.tau = 0.5

    def __call__(self, s, data, in_training = True):
        self._network.eval()
        out = self._network(s, data)
        probabilities = softmax(out, dim=-1)

        if in_training:
            m = Categorical(probabilities)
            action = m.sample()
        else:
            action = torch.argmax(out)

        return action.data.item()

    def update(self, s, data, a, gamma_t, delta):
        self._network.train()
        prob = self._network(s, data)
        log_prob = torch.log_softmax(prob, dim=-1)
        loss = -gamma_t * delta * log_prob

        self._optimizer.zero_grad()
        loss[a].backward()
        self._optimizer.step()

    def episode(self):
        pass


class Baseline(object):

    def __init__(self, b):
        self.b = b

    def __call__(self, s):
        return self.b

    def update(self, s, G):
        pass


class BaselineVApprox(object):

    def __init__(self, state_dimensions, learning_rate, network):

        self._state_dimensions = state_dimensions
        self._learning_rate = learning_rate
        self._network = network(state_dimensions, 1)
        self._optimizer = torch.optim.Adam(self._network.parameters(), learning_rate, [0.9, 0.999])

    def __call__(self, state):
        self._network.eval()
        return self.value(state).data

    def value(self, state):
        out = self._network(state)
        return out

    def update(self, state, G):
        self._network.train()
        vApprox = self.value(state)
        loss = (torch.tensor([G]) - vApprox[-1]) ** 2 / 2
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()


class Trajectory(object):
    def __init__(self, states, rewards, actions, value):
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.value = value

    def __lt__(self, other):
        return self.value < other.value

#reinforcement algorithm
class Reinforce(object):
    def __init__(self, env, gamma, pi, baseline):
        self._env = env
        self._gamma = gamma
        self._pi = pi
        self._baseline = baseline
        self.trajectory_memory = []
        self.memory_length = 4
        self.sum_rewards = []

    def generate_trajectory(self, in_training=True):
        self._env.reset()
        state = self._env.state()
        term = False
        states, rewards, actions = [], [0], []
        while not term:
            action = self._pi(state[0], state[1], in_training)
            term = self._env.take_action(action)
            next_state = self._env.state()
            next_reward = self._env.reward()
            states.append(state)
            rewards.append(next_reward)
            actions.append(action)
            state = next_state
            if len(states) > 20:
                term = True
        return Trajectory(states, rewards, actions, self._env.curr_state_value())

    def episode(self, in_training=True):
        trajectory = self.generate_trajectory(in_training=in_training)
        self.update_trajectory(trajectory, in_training)
        self._pi.episode()
        return self._env.returns()

    def update_trajectory(self, trajectory, in_training=True):
        states = trajectory.states
        rewards = trajectory.rewards
        actions = trajectory.actions
        bisect.insort(self.trajectory_memory, trajectory)
        self.seq_len = len(states)
        for t_idx in range(self.seq_len):
            G = sum(self._gamma ** (k - t_idx - 1) * rewards[k] for k in range(t_idx + 1, self.seq_len + 1))
            state = states[t_idx]
            action = actions[t_idx]
            baseline = self._baseline(state[0])
            delta = G - baseline
            self._baseline.update(state[0], G)
            self._pi.update(state[0], state[1], action, self._gamma ** t_idx, delta)
        self.sum_rewards.append(sum(rewards))
        print(sum(rewards))

    def replay(self):
        for idx in range(min(self.memory_length, int(len(self.trajectory_memory) / 10))):
            if len(self.trajectory_memory) / 10 < 1:
                return
            upper = min(len(self.trajectory_memory) / 10, 30)
            r1 = random.randint(0, upper)
            self.update_trajectory(self.trajectory_memory[idx])
