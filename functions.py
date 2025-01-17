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
    def __init__(self, num_actions, learning_rate, network):
        self._num_actions = num_actions
        self._learning_rate = learning_rate
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), learning_rate, [0.9, 0.999])
        self.tau = 0.5
    
    # Computes action preferences/probabilities using softmax, then either samples them using categorical distribustion (in_training = True) or selects the action with the highest probability (in_training = False)
    def __call__(self, s, data, in_training = True):
        self._network.eval() 
        out = self._network(s, data)
        probabilities = softmax(out, dim=-1)
        
        #print(probabilities)            
        
        if in_training:
            m = Categorical(probabilities)
            action = m.sample()
        else:
            action = torch.argmax(out)
            print(action)
            print(probabilities)

        return action.data.item()
    
    # Updates the network parameters based on the gradient of the policy objective
    def update(self, state_data, graph_data, action, gamma_t, delta):
        self._network.train()

        # Forward pass to get action probabilities
        prob = self._network(state_data, graph_data)
        log_prob = torch.log_softmax(prob, dim=-1)

        # Compute loss as per REINFORCE
        loss = -gamma_t * delta * log_prob[action]

        # Backpropagation
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

#    def episode(self):
#        pass


# Value function approximation class (a learned baseline); approximates the value of states using a neural network
class VApprox(object):
    
    # Initializes the baseline approximator by setting up a neural network with state dimensions as input and the value of the state as output
    def __init__(self, learning_rate, network):
        self._learning_rate = learning_rate
        self._network = network
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
