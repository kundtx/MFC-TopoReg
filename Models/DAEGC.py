import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from Models.GAT import GAT
from sklearn.preprocessing import normalize
import numpy as np

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial,requires_grad=True)



class DAEGC(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        # get pretrain model
        self.gat = GAT(num_features, hidden_size, embedding_size, alpha)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    
    def get_M(self, adj):
        # t_order
        t = 2
        tran_prob = torch.nn.functional.normalize(adj, p=1, dim=0)
        M_torch = sum([torch.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
        return M_torch.cuda()

    def forward(self, x, adj):
        M = self.get_M(adj)
        A_pred, z = self.gat(x, adj, M) # z has been normalized
        q = self.get_Q(z)

        return A_pred, z, q

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q