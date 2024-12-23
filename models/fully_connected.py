import numpy as np
import torch
from torch import nn

class FullyConnected(nn.Module):
    def __init__(self, input_features, hidden_features, output_features):
        super(FullyConnected, self).__init__()
        self._input_features = input_freatures
        self._output_features = output_features

        self.fully_connected_1 = nn.Linear(input_features, hidden_features)
        self.fully_connected_2 = nn.Linear(hidden_features, output_features)

    def forward(self, x):
        x = self.fully_connected_1(x)
        x = torch.relu(x)
        x = self.fully_connected_2(x)

        return x
