import abc_py
import numpy as np
from numpy import linalg
import torch
from torch_geometric.data import Data

#def symmetric_laplacian(abc):
#    num_nodes = abc.num_nodes()
#    lapalacian_matrix = np.zeros((num_nodes, num_nodes))
#    print("num_nodes", num_nodes)
#    
#    for node_idx in range(num_nodes):
#        aig_node = abc.aigNode(node_idx)
#        degree = float(aig_node.numFanouts())
#
#        if aig_node.hasFanin0():
#            degree += 1.0
#            fanin = aig_node.fanin0()
#            lapalacian_matrix[node_idx][fanin] = -1.0
#            lapalacian_matrix[fanin][node_idx] = -1.0
#
#        if aig_node.hasFanin1():
#            degree += 1.0
#            fanin = aig_node.fanin1()
#            lapalacian_matrix[node_idx][fanin] = -1.0
#            lapalacian_matrix[fanin][node_idx] = -1.0
#
#        lapalacian_matrix[node_idx][node_idx] = degree
#
#    return lapalacian_matrix
#
#def symmetric_lapalacian_eigen_values(abc):
#    lapalacian_matrix = symmetric_laplacian(abc)
#    print("L", lapalacian_matrix)
#
#    eigen_vals = np.real(linalg.eigvals(lapalacian_matrix))
#    print("eigVals", eigen_vals)
#
#    return eigen_vals

def extract_graph(abc):
    num_nodes = abc.numNodes()
    edge_index = []
    # Each node in the AIG is embedded with the following: [0 - 1 if constant1, 0 - 1 if PI, 0 - 1 if PO, 0 - 1 if fanin0 inverted, 0 - 1 if fanin1 inverted]
    features = torch.zeros((num_nodes, 6))
#       
#    for node_idx in range(num_nodes):
#        aig_node = abc.aigNode(node_idx)
#        node_type = aig_node.nodeType()  #  !Perhaps there are more specific node types?
#        features[node_idx][node_type] = 1.0
#
#        # Add edges to the edge_index list.
#        if aig_node.hasFanin0():
#            fanin = aig_node.fanin0()
#            edge_index.append([fanin, node_idx])
#        if aig_node.hasFanin1():
#            fanin = aig_node.fanin1()
#            edge_index.append([fanin, node_idx])
    
    for node_idx in range(num_nodes):
        aig_node = abc.aigNode(node_idx)
        node_type = aig_node.nodeType()  #  !Perhaps there are more specific node types?
        
        if node_type in [0, 1, 2]:
            features[node_idx][node_type] = 1.0
        else:
            features[node_idx][3] = 1.0
            if node_type == 3:
               features[node_idx][4] = 0.0
               features[node_idx][5] = 0.0
            elif node_type == 4:
               features[node_idx][4] = 1.0
               features[node_idx][5] = 0.0
            elif node_type == 5:
               features[node_idx][4] = 0.0
               features[node_idx][5] = 1.0
            elif node_type == 6:
               features[node_idx][4] = 1.0
               features[node_idx][5] = 1.0


        # Add edges to the edge_index list.
        if aig_node.hasFanin0():
            fanin = aig_node.fanin0()
            edge_index.append([fanin, node_idx])
        if aig_node.hasFanin1():
            fanin = aig_node.fanin1()
            edge_index.append([fanin, node_idx])

    # Convert edge_index to a tensor with shape [2, num_edges].
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # Create a PyTorch Geometric Data object.
    data = Data(x=features, edge_index=edge_index)

    return data
