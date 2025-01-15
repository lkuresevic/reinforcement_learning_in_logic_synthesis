import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data

class GCN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features):
        super(GCN, self).__init__()
        
        self.graph_conv_1 = SAGEConv(input_features, hidden_features)
        self.graph_conv_2 = SAGEConv(hidden_features, hidden_features)
        self.graph_conv_3 = SAGEConv(hidden_features, hidden_features)
        self.graph_conv_4 = SAGEConv(hidden_features, output_features)

    def forward(self, data):
        # `data` is expected to be a PyTorch Geometric Data object
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply graph convolutions with ReLU activations
        x = self.graph_conv_1(x, edge_index)
        x = torch.relu(x)
        x = self.graph_conv_2(x, edge_index)
        x = torch.relu(x)
        x = self.graph_conv_3(x, edge_index)
        x = torch.relu(x)
        x = self.graph_conv_4(x, edge_index)

        # Perform global mean pooling to aggregate node features for each graph
        hg = global_mean_pool(x, batch)

        return torch.squeeze(hg)
