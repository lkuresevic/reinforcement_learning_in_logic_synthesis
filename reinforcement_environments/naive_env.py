import abc_py
import numpy as np
import graph_feature_extractor as gfe
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class NaiveEnv(object):
    def __init__(self, aigfile):
        #
        self._abc = abc_py.AbcInterface()
        self._aigfile = aigfile

        #sequence lenght
        self.seq_len = 0
        
        #start Berkeley's ABC'
        self._abc.start()
        
        #read input AIG file
        self._abc.read(self._aigfile)
        
        #initial AIG statistics
        init_stats = self._abc.aigStats()  # The initial AIG statistics        
        self.init_numAnd = float(init_stats.numAnd)
        self.init_levels = float(init_stats.lev)

        self.resyn2()  # Run a compress2rs as target
        self.resyn2()
        resyn2_stats = self._abc.aigStats()

        total_reward = self.stat_value(init_stats) - self.stat_value(resyn2_stats)
        self._reward_baseline = total_reward / 20.0  # 18 is the length of compress2rs sequence
        print("baseline num AND ", resyn2_stats.numAnd, " total reward ", total_reward)

    def resyn2(self):
        self._abc.balance(l=False)
        self._abc.rewrite(l=False)
        self._abc.refactor(l=False)
        self._abc.balance(l=False)
        self._abc.rewrite(l=False)
        self._abc.rewrite(l=False, z=True)
        self._abc.balance(l=False)
        self._abc.refactor(l=False, z=True)
        self._abc.rewrite(l=False, z=True)
        self._abc.balance(l=False)

    def reset(self):
        self.seq_len = 0
        self._abc.end()
        self._abc.start()
        self._abc.read(self._aigfile)
        self._prev_stats = self._abc.aigStats()  # The initial AIG statistics
        self._curr_stats = self._lastStats  # The current AIG statistics
        self.prev_action_1 = self.numActions() - 1
        self.prev_action_2 = self.numActions() - 1
        self.prev_action_3 = self.numActions() - 1
        self.prev_action_4 = self.numActions() - 1
        self.actions_taken = np.zeros(self.numActions())

        return self.state()

    def close(self):
        self.reset()

    def step(self, actionIdx):
        self.take_action(actionIdx)
        next_state = self.state()
        reward = self.reward()
        done = False
        if self.seq_len >= 20:
            done = True
        return next_state, reward, done, 0

    def take_action(self, action_idx):
        """
        @return true: episode is end
        """
        self.prev_action_4 = self.prev_action_3
        self.prev_action_3 = self.prev_action_2
        self.prev_action_2 = self.prev_action_1
        self.prev_action_1 = action_idx
        self.seq_len += 1

        if action_idx == 0:
            self._abc.balance(l=False)  # b
        elif action_idx == 1:
            self._abc.rewrite(l=False)  # rw
        elif action_idx == 2:
            self._abc.refactor(l=False)  # rf
        elif action_idx == 3:
            self._abc.rewrite(l=False, z=True)  # rwz
        elif action_idx == 4:
            self._abc.refactor(l=False, z=True)  # rs
        elif action_idx == 5:
            self._abc.end()
            return True
        else:
            assert False

        # Update the statistics
        self._prev_stats = self._curr_stats
        self._curr_stats = self._abc.aigStats()
        return False

    def state(self):
        one_hot_activation = np.zeros(self.numActions())
        np.put(one_hot_activation, self.prev_action_1, 1)

        prev_one_hot_activations = np.zeros(self.numActions())
        prev_one_hot_activations[self.prev_activation_2] += 1 / 3
        prev_one_hot_activations[self.prev_activation_3] += 1 / 3
        prev_one_hot_activations[self.prev_activation_1] += 1 / 3
        prev_one_hot_activations = np.array([self._curr_stats.numAnd / self.init_numAnd, self._curr_stats.lev / self.init_levels, self._prev_stats.numAnd / self.init_numAnd, self._prev_stats.lev / self.init_levels])

        step_array = np.array([float(self.seq_len) / 20.0])
        combined = np.concatenate((state_array, prev_one_hot_activations, stepArray), axis=-1)
        return torch.from_numpy(combined.astype(np.float32)).float()

    def reward(self):
        if self.prev_action_1 == 5:  # Terminal
            return 0
        return self.state_value(self._prev_stats) - self.statValue(self._curr_stats) - self._reward_baseline

    def num_actions(self):
        return 5

    def state_dim(self):
        return 4 + self.num_actions() * 1 + 1

    def returns(self):
        return [self._curr_stats.numAnd, self._curr_stats.lev]

    def state_value(self, state):
        return float(state.numAnd) / float(self.init_numAnd)

    def curr_stateValue(self):
        return self.state_value(self._curr_stats)

    def seed(self, sd):
        pass

    def compress2rs(self):
        self._abc.compress2rs()
