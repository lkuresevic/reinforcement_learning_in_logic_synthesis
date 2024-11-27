import torch
from torch import nn
from torch_geometric.nn import *

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, out_len):
        super(GCN, self).__init__()
        pass

    def forward(self, g):
        pass