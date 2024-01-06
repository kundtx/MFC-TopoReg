import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import numpy as np
import time


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial,requires_grad=True)

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.leaky_relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)  # m k
        self.adj = adj
        self.activation = activation

    def forward(self, x):
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


class GAE(nn.Module):
    def __init__(self, adj, feature_dim, args): #adj here should be normalized
        super(GAE, self).__init__()
        self.v = 1
        # nodes * features --> m * n
        self.base_gcn = GraphConvSparse(feature_dim,  #feature dim
                                        args.encoded_space_dim, 
                                        adj)
        self.cluster_centroid = nn.Parameter(torch.Tensor(args.n_cluster, 
                                                          args.encoded_space_dim))
        torch.nn.init.xavier_normal_(self.cluster_centroid.data)
    
    def restart_clusters(self):
        torch.nn.init.xavier_normal_(self.cluster_centroid.data)

    def encode(self, _X):
        hidden_z = self.base_gcn(_X)  # m n
        return hidden_z
    
    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_centroid, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def forward(self, _input, _):
        # z = self.encode(_input)
        z = torch.nn.functional.normalize(self.encode(_input),dim=1)
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        q = self.get_Q(z)
        return A_pred, z, q
    