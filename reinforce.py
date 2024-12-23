import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import bisect
import random
from torch_geometric.nn import GCNConv
import torch_geometric

from models.gcn import *
from models.fully_connected import *
from models.fully_connected_graph import *

class PiApprox(object):

    def __init__(self, dimStates, numActs, alpha, network):

        self._dimStates = dimStates
        self._numActs = numActs
        self._alpha = alpha
        self._network = network(dimStates, numActs)
        self._optimizer = torch.optim.Adam(self._network.parameters(), alpha, [0.9, 0.999])
        self.tau = 0.5

    def __call__(self, s, data, phaseTrain=True):
        self._network.eval()
        out = self._network(s, data)
        probs = F.softmax(out, dim=-1)

        if phaseTrain:
            m = Categorical(probs)
            action = m.sample()
        else:
            action = torch.argmax(out)

        return action.data.item()

    def update(self, s, data, a, gammaT, delta):
        self._network.train()
        prob = self._network(s, data)
        logProb = torch.log_softmax(prob, dim=-1)
        loss = -gammaT * delta * logProb

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

    def __init__(self, dimStates, alpha, network):

        self._dimStates = dimStates
        self._alpha = alpha
        self._network = network(dimStates, 1)
        self._optimizer = torch.optim.Adam(self._network.parameters(), alpha, [0.9, 0.999])

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


class Reinforce(object):
    def __init__(self, env, gamma, pi, baseline):
        self._env = env
        self._gamma = gamma
        self._pi = pi
        self._baseline = baseline
        self.memTrajectory = []
        self.memLength = 4
        self.sumRewards = []

    def genTrajectory(self, phaseTrain=True):
        self._env.reset()
        state = self._env.state()
        term = False
        states, rewards, actions = [], [0], []
        while not term:
            action = self._pi(state[0], state[1], phaseTrain)
            term = self._env.takeAction(action)
            nextState = self._env.state()
            nextReward = self._env.reward()
            states.append(state)
            rewards.append(nextReward)
            actions.append(action)
            state = nextState
            if len(states) > 20:
                term = True
        return Trajectory(states, rewards, actions, self._env.curStatsValue())

    def episode(self, phaseTrain=True):
        trajectory = self.genTrajectory(phaseTrain=phaseTrain)
        self.updateTrajectory(trajectory, phaseTrain)
        self._pi.episode()
        return self._env.returns()

    def updateTrajectory(self, trajectory, phaseTrain=True):
        states = trajectory.states
        rewards = trajectory.rewards
        actions = trajectory.actions
        bisect.insort(self.memTrajectory, trajectory)
        self.lenSeq = len(states)
        for tIdx in range(self.lenSeq):
            G = sum(self._gamma ** (k - tIdx - 1) * rewards[k] for k in range(tIdx + 1, self.lenSeq + 1))
            state = states[tIdx]
            action = actions[tIdx]
            baseline = self._baseline(state[0])
            delta = G - baseline
            self._baseline.update(state[0], G)
            self._pi.update(state[0], state[1], action, self._gamma ** tIdx, delta)
        self.sumRewards.append(sum(rewards))
        print(sum(rewards))

    def replay(self):
        for idx in range(min(self.memLength, int(len(self.memTrajectory) / 10))):
            if len(self.memTrajectory) / 10 < 1:
                return
            upper = min(len(self.memTrajectory) / 10, 30)
            r1 = random.randint(0, upper)
            self.updateTrajectory(self.memTrajectory[idx])


