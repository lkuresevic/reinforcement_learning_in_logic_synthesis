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
from functions import *

    # Class holding the trajectory of states, rewards, actions and value for an episode
class Trajectory(object):

    # Stores the states, rewards, actions and the final value of the trajectory
    def __init__(self, states, rewards, actions, value):
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.value = value
    
    # Allows for sorting of trajectories based on the value of the trajectory
    def __lt__(self, other):
        return self.value < other.value

# Class representing the REINFORCE algorithm (a Monte Carlo policy gradient method)
class Reinforce(object):
    
    # Initializes the environment (env), discount factor (gamma), policy (pi) and baseline, as well as trajectory_memory
    def __init__(self, env, gamma, pi, baseline):
        self._env = env
        self._gamma = gamma
        self._pi = pi
        self._baseline = baseline
        self.trajectory_memory = []
        self.memory_length = 4
        self.sum_rewards = []
    
    # Generates an episode by interacting with the environment (self._env); collects states, rewards and actions until the episode terminates
    def generate_trajectory(self, state_dictionary, in_training=True):
        self._env.reset()
        state = self._env.state()
        term = False
        states, rewards, actions = [], [0], []

        while not term:
            action = self._pi(state[0], state[1], in_training)
            #print(f"action = {action}")
            
            if str(state) in state_dictionary:
                state_dictionary[str(state)] += 1
            else:
                state_dictionary[str(state)] = 1
            
            term = self._env.take_action(action)
            next_state = self._env.state()
            next_reward = self._env.reward()
            states.append(state)
            rewards.append(next_reward)
            actions.append(action)
            state = next_state
            if len(states) > 10:
                term = True
        
        return Trajectory(states, rewards, actions, self._env.curr_state_value())
    
    # Generates an episode and updates the policy and baseline using the collected trajectory
    def episode(self, state_dictionary, in_training=True):
        trajectory = self.generate_trajectory(state_dictionary, in_training=in_training)
        self.update_policy_and_baseline(trajectory, in_training)
        return self._env.returns()
    
    # Updates the policy and baseline based on the trajectory
    def update_policy_and_baseline(self, trajectory, in_training=True):
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
            print(f"{t_idx}: {G}, {baseline}")
            delta = G - baseline
            self._baseline.update(state[0], G)
            self._pi.update(state[0], state[1], action, self._gamma ** t_idx, delta)
        self.sum_rewards.append(sum(rewards))
        print("Rewards: " + str(sum(rewards)))
    
    # Replays a batch of stored trajectories from trajectory_memory, updating the policy and baseline based on these experiences
    def replay(self):
        for idx in range(min(self.memory_length, int(len(self.trajectory_memory) / 10))):
            if len(self.trajectory_memory) / 10 < 1:
                return
            upper = int(min(len(self.trajectory_memory) / 10, 30))
            r1 = random.randint(0, upper)
            self.update_policy_and_baseline(self.trajectory_memory[idx])
