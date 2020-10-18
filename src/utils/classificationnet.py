import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from src.utils.common import *


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, batchsize, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batchsize = batchsize
        self.weight = nn.Parameter(torch.FloatTensor(batchsize,in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(batchsize*out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.einsum("jik,ikp->jip",input,self.weight)
        support = torch.reshape(support,[support.size(0),-1])
        # support = torch.mm(input, self.weight)
        if self.bias is not None:
            support = support + self.bias
        output = torch.spmm(adj, support)
        output = torch.reshape(output,[output.size(0),self.batchsize,-1])
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, batchsize, dropout=0.5, softmax=True, bias=False):
        super(GCN, self).__init__()
        self.dropout_rate = dropout
        self.softmax = softmax
        self.batchsize = batchsize
        self.gc1 = GraphConvolution(nfeat, nhid, batchsize, bias=bias)
        self.gc2 = GraphConvolution(nhid, nclass, batchsize, bias=bias)
        self.dropout = nn.Dropout(p=dropout,inplace=False)

    def forward(self, x, adj):
        x = x.expand([self.batchsize]+list(x.size())).transpose(0,1)
        x = self.dropout(x)
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        x = x.transpose(0,1).transpose(1,2)
        if self.softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x.squeeze()

    def reset(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()