import torch
import torch.nn as nn

from models.gcn import *
from models.fully_connected import *

class FullyConnectedGraph(nn.Module):
    def __init__(self, input_features, hidden_features, output_features):
        super(FullyConnectedGraph, self).__init__()
        self._input_features = input_features
        self._output_features = output_features
        self.fully_connected_1 = nn.Linear(input_features, hidden_features-4)
        self.fully_connected_2 = nn.Linear(hidden_features, hidden_features)
        self.fully_connected_3 = nn.Linear(hidden_features, output_features)
        self.gcn = GCN(6, 128, 4)

    def forward(self, x, graph):
        graph_state = self.gcn(graph)

        x = self.fully_connected_1(x)
        x = torch.relu(x)
        x = self.fully_connected_2(torch.cat((x, graph_state), 0))
        x = torch.relu(x)
        x = self.fully_connected_3(x)

        return x
