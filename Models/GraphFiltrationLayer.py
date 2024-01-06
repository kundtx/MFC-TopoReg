import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import scipy.sparse as sp
from Code.dataloader import graph_normalization
####################################
# Vietoris-Rips filtration on Graph#
####################################


# The parameters of the model are the point coordinates.
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def build_community_graph(Q: torch.Tensor, W: sp.coo_matrix):
    # Q : the indicator matrix
    # W : the weighted graph adj matrix
    indicator = torch.argmax(Q, dim=1)
    indicator_hat = torch.stack(
        [torch.where(indicator == k, 1., 0.) for k in range(Q.size(1))]).T
    Q_hat = indicator_hat * Q
    if not torch.is_tensor(W):
        W = torch.tensor(W.todense(), dtype=torch.float, device="cuda:0")
    result = torch.mm(torch.mm(Q_hat.T, W), Q_hat).fill_diagonal_(0)
    result = result/ W.sum()
    return result


def wrcf(G: nx.Graph):  # networkx version
    """Compute the weight-rank clique filtration (WRCF) of a graph.
    :param G: networkx Graph
    :param weight: name of the weight attribute
    :return: a gudhi filtration.
    """
    # Define filtration step 0 as the set of all nodes
    st = gd.SimplexTree()
    for v in G.nodes():
        st.insert([v], filtration=0)
    # Rank all edge weights (from large to small)
    distinct_weights = np.unique([i[2] for i in G.edges.data("weight")])[::-1]
    for t, w in enumerate(distinct_weights):
        # At filtration step t, threshold the graph at weight[t]
        subg = G.edge_subgraph([(u, v) for u, v, _w, in G.edges.data("weight")
                                if _w >= w])
        # Find all maximal cliques and define them to be simplices
        for clique in nx.find_cliques(subg):
            # st.insert(clique,filtration=t+1) # the version used in thesis
            st.insert(clique, filtration=1 / w)
    return st


def WRCF_Index(G, dim, card):
    # Parameters: G (adjancy matrix),
    #             dim (homological dimension),
    #             card (number of persistence diagram points, sorted by distance-to-diagonal)

    # Compute the persistence pairs with Gudhi

    st = wrcf(nx.from_numpy_matrix(G))
    dgm = st.persistence()
    pairs = st.persistence_pairs()

    # Retrieve vertices v_a and v_b by picking the ones achieving the maximal
    # distance among all pairwise distances between the simplex vertices
    indices, pers = [], []
    for s1, s2 in pairs:
        if len(s1) == dim + 1 and len(s2) > 0:
            l1, l2 = np.array(s1), np.array(s2)
            i1 = [
                s1[v] for v in np.unravel_index(np.argmax(G[l1, :][:, l1]),
                                                [len(s1), len(s1)])
            ]
            i2 = [
                s2[v] for v in np.unravel_index(np.argmax(G[l2, :][:, l2]),
                                                [len(s2), len(s2)])
            ]
            indices += i1
            indices += i2
            pers.append(st.filtration(s2) - st.filtration(s1))

    # Sort points with distance-to-diagonal
    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1, 4])[perm][::-1, :].flatten())
    # [perm] sort, [::-1,:] reverse

    # Output indices
    indices = indices[:4 * card] + [
        0 for _ in range(0, max(0, 4 * card - len(indices)))
    ]
    return np.array(indices, dtype=np.int32)


class WrcfLayer(torch.nn.Module):

    def __init__(self, dim=1, card=50):
        super(WrcfLayer, self).__init__()
        self.dim = dim
        self.card = card

    def forward(self, G: torch.Tensor):
        d, c = self.dim, self.card
        G = G.cpu()
        # Compute vertices associated to positive and negative simplices
        # Don't compute gradient for this operation
        with torch.no_grad():
            ids = torch.from_numpy(WRCF_Index(G.numpy(), d, c))  # from_numpy will keep dtype
        # Get persistence diagram by simply picking the corresponding entries in the distance matrix
        if d > 0:
            indices = ids.view([2 * c, 2]).long()
            dgm = G[indices[:, 0], indices[:, 1]].view(c, 2)
        else:
            indices = ids.view([2 * c, 2])[1::2, :].long()
            dgm = torch.cat(
                [
                    torch.zeros(c, 1),  # birth will always be zero
                    G[indices[:, 0], indices[:, 1]].view(c, 1).float()
                ],
                dim=1)
        return dgm.cuda()


class TopoLoss(torch.nn.Module):

    def __init__(self, nearby_dgms, args) -> None:
        super().__init__()
        self.wrcf_layer_dim0 = WrcfLayer(dim=0, card=args.card)
        self.wrcf_layer_dim1 = WrcfLayer(dim=1, card=args.card)
        self.dgm_gt = nearby_dgms
        self.LAMBDA = args.LAMBDA

    def forward(self, adj, soft_label):

        # compute graph filtration based topological loss
        C = build_community_graph(soft_label, adj)
        dgm_dim0 = self.wrcf_layer_dim0(C)
        dgm_dim1 = self.wrcf_layer_dim1(C)

        # loss_topo = torch.square(wasserstein_distance(dgm, self.dgm_gt, order=2,
        #                 enable_autodiff=True, keep_essential_parts=False))
        if self.dgm_gt[0]:
            topo_dim0_before = wasserstein_distance(dgm_dim0,
                                            self.dgm_gt[0][0],
                                            order=1,
                                            # internal_p=2,
                                            enable_autodiff=True,
                                            keep_essential_parts=False)
            topo_dim1_before = wasserstein_distance(dgm_dim1,
                                            self.dgm_gt[0][1],
                                            order=1,
                                            # internal_p=2,
                                            enable_autodiff=True,
                                            keep_essential_parts=False)
        if self.dgm_gt[1]:
            topo_dim0_next = wasserstein_distance(dgm_dim0,
                                            self.dgm_gt[1][0],
                                            order=1,
                                            # internal_p=2,
                                            enable_autodiff=True,
                                            keep_essential_parts=False)
            

            topo_dim1_next = wasserstein_distance(dgm_dim1,
                                            self.dgm_gt[1][1],
                                            order=1,
                                            # internal_p=2,
                                            enable_autodiff=True,
                                            keep_essential_parts=False)
        
        if not self.dgm_gt[0]:
            loss_topo = topo_dim0_next + topo_dim1_next
        elif not self.dgm_gt[1]:
            loss_topo = topo_dim0_before + topo_dim1_before
        else:
            loss_topo = topo_dim0_before + topo_dim1_before + topo_dim0_next + topo_dim1_next
        return  self.LAMBDA * loss_topo
