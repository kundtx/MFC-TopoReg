from gudhi.wasserstein import wasserstein_distance
import torch
import torch.nn.functional as F
from Code.dataloader import sparse_to_tuple, graph_normalization
from Models import *
from .utils import *
import time
from Code.evaluation import get_acc
from Models.GraphFiltrationLayer import TopoLoss

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def base_train(network_type,model,features,adj,args,idx):
    if network_type == "DAEGC":
        daegc_trainer(model,features,adj,args,None,idx)
    elif network_type == "GEC":
        gae_trainer(model,features,adj,args,None,idx)
    elif network_type == "MFC":
        gaemf_trainer(model,features,adj,args,None,idx)
    elif network_type == "SDCN":
        sdcn_trainer(model,features,adj,args,None,idx)
    else:
        raise ValueError(f'Unknown network type: {network_type}')
    
def retrain_with_topo(
        network_type,
        _model,
        dgm_gt,
        adj,
        features,
        args,
        idx):
    topo_loss = TopoLoss(nearby_dgms=dgm_gt, args=args)
    # train model
    if network_type == "DAEGC":
        daegc_trainer(_model,features,adj,args,topo_loss,idx)
    elif network_type == "GEC":
        gae_trainer(_model,features,adj,args,topo_loss,idx)
    elif network_type == "MFC":
        gaemf_trainer(_model,features,adj,args,topo_loss,idx)
    elif network_type == "SDCN":
        sdcn_trainer(_model,features,adj,args,topo_loss,idx)
    else:
        raise ValueError(f'Unknown network type: {network_type}')
    
def gae_trainer(
        model:GAE,
        features:torch.Tensor,
        adj:sp.coo_matrix,
        args:dict,
        topo:TopoLoss,
        idx:str
    ):
    device = "cuda"
    adj_label = sparse_to_tuple(adj)
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                        torch.FloatTensor(adj_label[1]),
                                        torch.Size(adj_label[2])).to(device)


    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight

    if topo:
        optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.learning_rate,
                                  weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=0.001,
                                  weight_decay=1e-4)

    # train model
    model.train()
    for epoch in range(args.num_epoch):
        t = time.time()
        if epoch % 1 == 0:
            # update_interval
            A_pred, z, Q = model(features,adj)
        A_pred, z, q = model(features,adj)
        optimizer.zero_grad()
        re_loss = norm * F.binary_cross_entropy(A_pred.view(-1),
                                                 adj_label.to_dense().view(-1),
                                                 weight=weight_tensor)
        
        if topo:
            loss = topo(adj,q) + re_loss
        else:
            p = target_distribution(Q.detach())
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
            loss = 10 * kl_loss + re_loss

        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            if int(idx) >= 9 and args.file_name == "Data/Cora":
                train_acc = 0
            else:
                train_acc = get_acc(A_pred, adj_label.to_dense())
            print("Epoch:", '%04d' % (epoch + 1), 
              "extra_loss=", "{:.5f}".format(loss.item() - re_loss.item()),
              "re_loss=", "{:.5f}".format(re_loss.item()),
              "train_acc=", "{:.5f}".format(train_acc), 
              "time=", "{:.5f}".format(time.time() - t))
        
def daegc_trainer(
        model:DAEGC,
        features:torch.Tensor,
        adj:torch.Tensor,
        args:dict, 
        topo:TopoLoss,
        idx:str
    ):
    optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=args.learning_rate, 
                                    weight_decay=1e-4)
    adj_label = adj.clone()
    adj = graph_normalization(adj)
    # get kmeans and pretrain cluster result
    model.train()
    update_interval = 1 #1,3,5
    for epoch in range(args.num_epoch):
        t = time.time()
        if epoch % update_interval == 0:
            # update_interval
            A_pred, z, Q = model(features, adj)

        A_pred, z, q = model(features, adj)
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))

        if topo:
            loss = topo(adj,q) + re_loss
        else:
            p = target_distribution(Q.detach())
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
            loss = 10 * kl_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            if int(idx)>=9 and args.file_name == "Data/Cora":
                train_acc = 0
            else:
                train_acc = get_acc(A_pred, adj_label.to_dense())
            print("Epoch:", '%04d' % (epoch + 1), 
              "extra_loss=", "{:.5f}".format(loss.item() - re_loss.item()),
              "re_loss=", "{:.5f}".format(re_loss.item()),
              "train_acc=", "{:.5f}".format(train_acc), 
              "time=", "{:.5f}".format(time.time() - t))
        
def gaemf_trainer(
        model:GAEMF,
        features:torch.Tensor,
        adj:torch.Tensor,
        args:dict, 
        topo:TopoLoss,
        idx:str
    ):
    device = "cuda"
    adj_label = sparse_to_tuple(adj)
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                        torch.FloatTensor(adj_label[1]),
                                        torch.Size(adj_label[2])).to(device)


    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight

    if topo:
        optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.learning_rate,
                                  weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.learning_rate,
                                  weight_decay=1e-4)
    model.train()
    for epoch in range(args.num_epoch):
        t = time.time()
        mf_flag = epoch>args.start_mf
        if topo:
            mf_flag = True
        A_pred, z, q = model(features, mf_flag)
        re_loss = norm * F.binary_cross_entropy(A_pred.view(-1),
                                                 adj_label.to_dense().view(-1),
                                                 weight=weight_tensor)

        if topo:
            loss = topo(adj,q) + re_loss
        elif epoch>args.start_mf:
            loss_kmeans = F.mse_loss(z, torch.mm(q, model.cluster_centroid))
            loss = 1 * loss_kmeans + re_loss
        else:
            loss = re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            if int(idx)>=9 and args.file_name == "Data/Cora":
                train_acc = 0
            else:
                train_acc = get_acc(A_pred, adj_label.to_dense())
            print("Epoch:", '%04d' % (epoch + 1), 
              "extra_loss=", "{:.5f}".format(loss.item() - re_loss.item()),
              "re_loss=", "{:.5f}".format(re_loss.item()),
              "train_acc=", "{:.5f}".format(train_acc), 
              "time=", "{:.5f}".format(time.time() - t))
        
def sdcn_trainer(
        model:SDCN,
        features:torch.Tensor,
        adj:torch.Tensor,
        args:dict, 
        topo:TopoLoss,
        idx:str
    ):
    adj = graph_normalization(adj)
    # features = torch.eye(adj.shape[0]).cuda()
    # get kmeans and pretrain cluster result
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epoch):
        t = time.time()
        if epoch % 1 == 0:
        # update_interval
            _, tmp_q, pred, _ = model(features, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
    
        x_bar, q, pred, _ = model(features, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, features.to_dense())
        if topo:
            loss = topo(adj,q) + re_loss
        else:
            # loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss
            loss = 1 * kl_loss + 1 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch:", '%04d' % (epoch + 1), 
            "extra_loss=", "{:.5f}".format(loss.item() - re_loss.item()),
            "re_loss=", "{:.5f}".format(re_loss.item()),
            "time=", "{:.5f}".format(time.time() - t))