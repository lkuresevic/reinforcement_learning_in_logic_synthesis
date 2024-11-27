import torch
from torch import nn
from torch_geometric.nn import *

class FullyConnectedGraph(nn.Module):
    def __init__(self, numFeats, outChs):
        super(FcModelGraph, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs
        self.fc1 = nn.Linear(numFeats, 32-4)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, outChs)
        self.gcn = GCN(6, 12, 4)

    def forward(self, x, graph):
        graph_state = self.gcn(graph)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(torch.cat((x, graph_state), 0))
        x = self.act2(x)
        x = self.fc3(x)
        return x