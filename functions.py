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
from reinforce import *

# Policy approximation class; a neural network takes states as inputs and outputs preference/probabilities for the actions
class PiApprox(object):
    
    # Initializes an object that is essentialy a neural network model
    def __init__(self, state_dimensions, num_actions, learning_rate, network):
        self._state_dimensions = state_dimensions
        self._num_actions = num_actions
        self._learning_rate = learning_rate
        self._network = network(state_dimensions, 128, num_actions)
        self._optimizer = torch.optim.Adam(self._network.parameters(), learning_rate, [0.9, 0.999])
        self.tau = 0.5
    
    # Computes action preferences/probabilities using softmax, then either samples them using categorical distribustion (in_training = True) or selects the action with the highest probability (in_training = False)
    def __call__(self, s, data, in_training = True):
        self._network.eval() 
        out = self._network(s, data)
        probabilities = softmax(out, dim=-1)
        
        print(probabilities)            
        
        if in_training:
            m = Categorical(probabilities)
            action = m.sample()
        else:
            action = torch.argmax(out)
        
        print(action)
        return action.data.item()
    
    # Updates the network parameters based on the gradient of the policy objective
    def update(self, s, data, a, gamma_t, delta):
        self._network.train()
        prob = self._network(s, data)
        loss = -gamma_t * delta * torch.log_softmax(prob, dim=-1)

        self._optimizer.zero_grad()
        loss[a].backward()
        self._optimizer.step()

#    def episode(self):
#        pass


# Class represents a baseline used for variance reduction in the policy gradient
class Baseline(object):
    
    # Initializes a baseline value
    def __init__(self, b):
        self.b = b
    
    # Returns the baseline value
    def __call__(self, s):
        return self.b
    
    # Updates the baseline using the recieved return G
    def update(self, s, G): 
        pass

# Value function approximation class (a learned baseline); approximates the value of states using a neural network
class BaselineVApprox(object):
    
    # Initializes the baseline approximator by setting up a neural network with state dimensions as input and the value of the state as output
    def __init__(self, state_dimensions, learning_rate, network):

        self._state_dimensions = state_dimensions
        self._learning_rate = learning_rate
        self._network = network(state_dimensions, 128, 1)
        self._optimizer = torch.optim.Adam(self._network.parameters(), learning_rate, [0.9, 0.999])

    # Returns the estimated value of a given state by passing it through the network
    def __call__(self, state):
        self._network.eval()
        return self.value(state).data
    
    # Approximates the state value
    def value(self, state):
        out = self._network(state)
        return out
    
    # Computes the mean squared error between the predicted value vApprox and the actual return G, then performs backpropagation to update the network parameters
    def update(self, state, G):
        self._network.train()
        vApprox = self.value(state)
        loss = (torch.tensor([G]) - vApprox[-1]) ** 2 / 2
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
