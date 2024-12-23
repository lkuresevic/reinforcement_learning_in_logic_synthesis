import torch
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_length):
        super(GCN, self).__init__()
        self.graph_conv_1 = GCNConv(input_features, hidden_features)
        self.graph_conv_2 = GCNConv(hidden_features, hidden_features)
        self.graph_conv_3 = GCNConv(hidden_features, hidden_features)
        self.graph_conv_4 = GCNConv(hidden_features, output_length)

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
