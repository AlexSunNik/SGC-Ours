import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)

class SGC_Big(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass, nhid):
        super(SGC_Big, self).__init__()
        self.W1 = nn.Linear(nfeat, nhid)
        self.W2 = nn.Linear(nhid, nclass)
        self.dropout = 0.9

    def forward(self, x, use_relu=True):
        x = self.W1(x)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.W2(x)
        return x

class SGC_Big_CS(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass, nhid):
        super(SGC_Big, self).__init__()
        self.W1 = nn.Linear(nfeat, nhid)
        self.W2 = nn.Linear(nhid, nclass)
        self.dropout = 0.9

    def forward(self, x, use_relu=True):
        x = self.W1(x)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.W2(x)
        return x

class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.W(input)
        # print(support.shape, adj.shape)
        output = torch.spmm(adj, support)
        return output

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

def get_model(model_opt, nfeat, nclass, nhid=0, dropout=0, cuda=True, num_edges=None):
    if model_opt == "GCN":
        model = GCN(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=dropout)
    elif model_opt in ["SGC"]:
        model = SGC(nfeat=nfeat,
                    nclass=nclass)
    elif model_opt in ["SGC-Concat"]:
        model = SGC_Big(nfeat=nfeat, nclass=nclass, nhid=nhid)
    elif model_opt in ["SGC-LPA"]:
        model = SGC_LPA(in_feature=nfeat, out_feature=nclass, hidden=nhid, dropout=0.9, num_edges=num_edges, lpaiters=2)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if cuda: model.cuda()
    return model

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from typing import Callable, Optional
from torch_geometric.nn import SGConv
# ************************************************************************* #
# Add LPA
class LPAconv(MessagePassing):
    def __init__(self, num_layers: int):
        super(LPAconv, self).__init__(aggr='add')
        self.num_layers = num_layers

    def forward(
            self, y: Tensor, edge_index: Adj, mask: Optional[Tensor] = None,
            edge_weight: OptTensor = None,
            post_step: Callable = lambda y: y.clamp_(0., 1.)
    ) -> Tensor:

        if y.dtype == torch.int64:
            y = F.one_hot(y.view(-1)).to(torch.float)

        out = y
        if mask is not None:
            out = torch.zeros_like(y)
            out[mask] = y[mask]

        if isinstance(edge_index, SparseTensor) and not edge_index.has_value():
            edge_index = gcn_norm(edge_index, add_self_loops=False)
        elif isinstance(edge_index, Tensor) and edge_weight is None:
            edge_index, edge_weight = gcn_norm(edge_index, num_nodes=y.size(0),
                                               add_self_loops=False)

        for _ in range(self.num_layers):
            # propagate_type: (y: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight,
                                 size=None)
            # out = post_step(out)
        return out


class SGC_LPA(nn.Module):
    def __init__(self, in_feature, hidden, out_feature, dropout, num_edges, lpaiters):
        super(SGC_LPA, self).__init__()
        self.edge_weight = nn.Parameter(torch.ones(num_edges))
        self.conv1 = SGConv(in_feature, out_feature, K=2, cached=True)
        self.lpa = LPAconv(lpaiters)
        self.dropout_rate = dropout

    def forward(self, x, adj, y, mask):
        x, edge_index, y = data.x, data.edge_index, data.y
        x = self.conv1(x, adj, edge_weight=self.edge_weight)
        y_hat = self.lpa(y, edge_index, mask, self.edge_weight)

        return x, y_hat