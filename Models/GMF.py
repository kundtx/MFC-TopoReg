import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from Models.GAT import GAT
from Models.GAE import GraphConvSparse,glorot_init

class GAEMF(nn.Module):
    def __init__(self, adj, feature_dim, args): #adj here should be normalized
        super(GAEMF, self).__init__()
        # nodes * features --> m * n
        self.start_mf = args.start_mf
        self.base_gcn = GraphConvSparse(feature_dim,  #feature dim
                                        args.encoded_space_dim, 
                                        adj)
        self.cluster_centroid =  glorot_init(args.n_cluster, args.encoded_space_dim)
    
    def restart_clusters(self):
        torch.nn.init.xavier_normal_(self.cluster_centroid.data)

    def encode(self, _X):
        hidden_z = self.base_gcn(_X)  # m n
        return hidden_z
    
    @staticmethod
    def normalize(X):
        X_std = (X - X.min(dim=1).values[:, None]) / (X.max(dim=1).values - X.min(dim=1).values)[:, None]
        return X_std / torch.sum(X_std, dim=1)[:, None]

    def forward(self, _input, flag):   
        z = self.encode(_input) # do not normalize z in MFC
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        if type(flag) != bool or flag is True:
            pinv_weight = torch.linalg.pinv(self.cluster_centroid)  # compute pesudo inverse of W [ n * k ]

            indicator = self.normalize(torch.mm(z, pinv_weight))  # m * n --> m * k
            # indicator = F.softmax(torch.mm(z, pinv_weight), dim=1)  # m * n --> m * k
            return A_pred, z, indicator
        else:
            return A_pred, z, None