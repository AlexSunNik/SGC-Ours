import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed, sgc_precompute_concat
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter
from torch_geometric.nn import CorrectAndSmooth
from torch_sparse import SparseTensor

# Arguments
args = get_citation_args()

if args.tuned:
    if args.model in["SGC", "SGC-Concat", "SGC-LPA"]:
        with open("{}-tuning/{}.txt".format("SGC", args.dataset), 'rb') as f:
            args.weight_decay = pkl.load(f)['weight_decay']
            print("using tuned weight decay: {}".format(args.weight_decay))
    else:
        raise NotImplemented

# setting random seeds
set_seed(args.seed, args.cuda)

# adj, features, labels, idx_train, idx_val, idx_test, num_edges = load_citation(args.dataset, args.normalization, args.cuda)
adj, features, labels, idx_train, idx_val, idx_test, num_edges, DAD, DA = load_citation(args.dataset, args.normalization, args.cuda)
if args.model in["SGC-Concat"]:
    model = get_model(args.model, features.size(1) * (args.degree+1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)
elif args.model in ["SGC-LPA"]:
    model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda, num_edges=num_edges)
else:
    model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)
# print(adj.shape, features.shape)

if args.model in["SGC"]:
    features, precompute_time = sgc_precompute(features, adj, args.degree)
    print("{:.4f}s".format(precompute_time))
elif args.model in["SGC-Concat"]:
    features, precompute_time = sgc_precompute_concat(features, adj, args.degree)
    print("{:.4f}s".format(precompute_time))

def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout, need_adj_fwd=False, adj=None, val_adj=None, all_features=None, post_cs=None, train_idx=None, DAD=None, DA=None, val_idx=None, test_idx=None, test_labels=None):

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # print(train_features.shape)
        if not need_adj_fwd:
            output = model(train_features)
        else:
            output = model(train_features, adj=adj)
        loss_train = F.cross_entropy(output, train_labels)
        # print("Train Loss", loss_train.item())
        loss_train.backward()
        optimizer.step()
        # with torch.no_grad():
        #     model.eval()
        #     output = model(val_features)
        #     acc_val = accuracy(output, val_labels)
        #     print("Val Acc", acc_val.item())
    train_time = perf_counter()-t
    model.eval()
    if all_features is not None and post_cs is not None:
        with torch.no_grad():
            output = model(all_features)
        y_soft = output.softmax(dim=-1)
        # print(type(y_soft), type(train_labels), type(train_idx), type(DAD.to_dense()))
        # print(train_labels.device, y_soft.device, train_idx.device, DAD.device)
        # print(y_soft.shape, train_labels.shape, train_idx.shape)
        DAD = DAD.to_dense()
        indices = torch.nonzero(DAD).t()
        DAD = SparseTensor(row=indices[0], col=indices[1], value=DAD[indices[0], indices[1]], sparse_sizes=DAD.size())

        DA = DA.to_dense()
        indices = torch.nonzero(DA).t()
        DA = SparseTensor(row=indices[0], col=indices[1], value=DA[indices[0], indices[1]], sparse_sizes=DA.size())
        # values = DAD[indices[0], indices[1]]
        # DAD = torch.sparse.FloatTensor(indices, values, DAD.size())
        y_soft = post_cs.correct(y_soft=y_soft, y_true=train_labels.unsqueeze(-1), mask=train_idx, edge_index=DAD)
        # y_soft = post_cs.smooth(y_soft=y_soft, y_true=train_labels.unsqueeze(-1), mask=train_idx, edge_index=DA)
        y_soft = post_cs.smooth(y_soft=y_soft, y_true=train_labels.unsqueeze(-1), mask=train_idx, edge_index=DAD)
        # print(y_soft)
        acc_val = accuracy(y_soft[val_idx], val_labels)
        acc_test = accuracy(y_soft[test_idx], test_labels)
    # with torch.no_grad():
    #     model.eval()
    #     if not need_adj_fwd:
    #         output = model(val_features)
    #     else:
    #         output = model(val_features, adj=val_adj)
    #     acc_val = accuracy(output, val_labels)

    return model, acc_val, train_time, acc_test

def test_regression(model, test_features, test_labels, need_adj_fwd=False, adj=None):
    model.eval()
    if not need_adj_fwd:
        return accuracy(model(test_features), test_labels)
    else:
        return accuracy(model(test_features, adj), test_labels)

train_adj = adj.to_dense()[idx_train]
train_adj = train_adj[:, idx_train]
indices = torch.nonzero(train_adj).t()
values = train_adj[indices[0], indices[1]]
train_adj = torch.sparse.FloatTensor(indices, values, train_adj.size())

val_adj = adj.to_dense()[idx_val]
val_adj = val_adj[:, idx_val]
indices = torch.nonzero(val_adj).t()
values = val_adj[indices[0], indices[1]]
val_adj = torch.sparse.FloatTensor(indices, values, val_adj.size())

test_adj = adj.to_dense()[idx_test]
test_adj = test_adj[:, idx_test]
indices = torch.nonzero(test_adj).t()
values = test_adj[indices[0], indices[1]]
test_adj = torch.sparse.FloatTensor(indices, values, test_adj.size())
# print("testing")
# print(train_adj.shape)
# train_adj = torch.sparse.FloatTensor(train_adj)
# post_cs = CorrectAndSmooth(num_correction_layers=5, correction_alpha=1, num_smoothing_layers=5, smoothing_alpha=0.8, autoscale=False, scale=20.)
post_cs = CorrectAndSmooth(num_correction_layers=2, correction_alpha=0.2, num_smoothing_layers=2, smoothing_alpha=0.2, autoscale=True)
if args.model in ["SGC", "SGC-Concat"]:
    model, acc_val, train_time, acc_test = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                     args.epochs, args.weight_decay, args.lr, args.dropout, all_features=features, post_cs=post_cs, train_idx=idx_train, DAD=DAD, DA=DA, val_idx=idx_val, test_idx=idx_test, test_labels=labels[idx_test])
    # acc_test = test_regression(model, features[idx_test], labels[idx_test])
elif args.model == "GCN":
    model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                     args.epochs, args.weight_decay, args.lr, args.dropout, True, train_adj, val_adj)
    acc_test = test_regression(model, features[idx_test], labels[idx_test], True, test_adj)
    precompute_time = 0
print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))
