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
                     lr=args.lr, dropout=args.dropout, need_adj_fwd=False, adj=None, val_adj=None):

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

    with torch.no_grad():
        model.eval()
        if not need_adj_fwd:
            output = model(val_features)
        else:
            output = model(val_features, adj=val_adj)
        acc_val = accuracy(output, val_labels)

    return model, acc_val, train_time

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
if args.model in ["SGC", "SGC-Concat"]:
    model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                     args.epochs, args.weight_decay, args.lr, args.dropout)
    acc_test = test_regression(model, features[idx_test], labels[idx_test])
elif args.model == "GCN":
    model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                     args.epochs, args.weight_decay, args.lr, args.dropout, True, train_adj, val_adj)
    acc_test = test_regression(model, features[idx_test], labels[idx_test], True, test_adj)
    precompute_time = 0
print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))
