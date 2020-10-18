import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.utils.const import MIN_EPSILON,MAX_EXP


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, batchsize, bias=False):

        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batchsize = batchsize
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def forward(self, input, adj):

        support = torch.einsum("jik,kp->jip",input,self.weight)
        if self.bias is not None:
            support = support + self.bias
        support = torch.reshape(support,[support.size(0),-1])
        output = torch.spmm(adj, support)
        output = torch.reshape(output,[output.size(0),self.batchsize,-1])
        return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# vanilla GCN
class PolicyNet(nn.Module):

    def __init__(self,args,statedim):

        super(PolicyNet, self).__init__()
        self.args = args
        self.gcn = nn.ModuleList()
        for i in range(len(args.pnhid)):
            if (i == 0):
                self.gcn.append(GraphConvolution(statedim, args.pnhid[i], args.batchsize, bias=True))
            else:
                self.gcn.append(GraphConvolution(args.pnhid[i - 1], args.pnhid[i], args.batchsize, bias=True))

        self.output_layer = nn.Linear(args.pnhid[-1], 1, bias=False)


    def forward(self, state, adj):

        x = state.transpose(0, 1)
        for layer in self.gcn:
            x = F.relu(layer(x, adj))
        x = self.output_layer(x).squeeze(-1).transpose(0, 1)
        return x


class PolicyNet2(nn.Module):

    def __init__(self, args, statedim):
        super(PolicyNet2,self).__init__()
        self.args = args
        self.lin1 = nn.Linear(statedim,args.pnhid[0])
        self.lin2 = nn.Linear(args.pnhid[0],args.pnhid[0])
        self.lin3 = nn.Linear(args.pnhid[0], 1)
        stdv = 1. / math.sqrt(self.lin1.weight.size(1))

        self.lin1.weight.data.uniform_(-stdv, stdv)
        self.lin2.weight.data.uniform_(-stdv, stdv)
        self.lin3.weight.data.uniform_(-stdv, stdv)

        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.zeros_(self.lin2.bias)
        torch.nn.init.zeros_(self.lin3.bias)


    def forward(self,state,adj):
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))
        logits = self.lin3(x).squeeze(-1)
        return logits


'''
# concatenate node features and graph encoding
class PolicyNet(nn.Module):

    def __init__(self,args,statedim):
        super(PolicyNet, self).__init__()
        self.args = args
        self.gc1 = GraphConvolution(statedim, args.pnhid, args.batchsize, bias=True)
        self.gc2 = GraphConvolution(args.pnhid, args.pnhid2, args.batchsize, bias=True)
        self.lin = nn.Linear(args.pnhid2, 1, bias=True)
        self.dropout=nn.Dropout(p=args.pdropout,inplace=False)

    def forward(self, state, adj):
        x = F.relu(self.gc1(state.transpose(0,1), adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        x = F.relu(x)
        x = self.lin(x)
        x = x.squeeze(-1).transpose(0,1)
        return x
'''
'''
# use graph encoding as an attention query on node features
class PolicyNet(nn.Module):

    def __init__(self,args,statedim):
        super(PolicyNet, self).__init__()
        self.args = args
        self.gc1 = GraphConvolution(statedim - args.use_degree_distribution, args.pnhid, args.batchsize, bias=True)
        self.gc2 = GraphConvolution(args.pnhid, args.pnhid2, args.batchsize, bias=True)
        self.dropout=nn.Dropout(p=args.pdropout, inplace=False)

        if (self.args.use_degree_distribution == 0):
            self.output_layer = nn.Linear(args.pnhid2, 1, bias=False)
        else:
            #self.graph_embed = nn.Linear(args.use_degree_distribution, args.pnhid2, bias=True)
            self.graph_embed = nn.Linear(args.use_degree_distribution, 2 * args.pnhid2, bias=True)
            self.graph_embed_2 = nn.Linear(2 * args.pnhid2, args.pnhid2, bias=True)

    def forward(self, state, adj):
        if (self.args.use_degree_distribution != 0):
            graph_summary = state[:, :, -self.args.use_degree_distribution:]
            state = state[:, :, :-self.args.use_degree_distribution]

        x = F.relu(self.gc1(state.transpose(0, 1), adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        x = F.relu(x)

        if (self.args.use_degree_distribution == 0):
            x = self.output_layer(x).squeeze(-1).transpose(0, 1)
            #x = 10. * torch.tanh(x)
        else:
            #graph_summary = self.graph_embed(graph_summary)
            graph_summary = F.relu(self.graph_embed(graph_summary))
            graph_summary = self.graph_embed_2(graph_summary)
            x = (x.transpose(0, 1) * graph_summary).sum(dim=2)
        
        return x
'''
'''
# use history pooling as global encoding
class PolicyNet(nn.Module):

    def __init__(self,args,statedim):
        super(PolicyNet, self).__init__()
        self.args = args
        self.gc1 = GraphConvolution(statedim, args.pnhid, args.batchsize, bias=True)
        self.gc2 = GraphConvolution(args.pnhid, args.pnhid2, args.batchsize, bias=True)
        self.dropout=nn.Dropout(p=args.pdropout, inplace=False)

        #self.global_embed = nn.Linear(statedim - 1, args.pnhid2, bias=True)
        self.global_rnn = nn.LSTM(input_size=statedim - 1, hidden_size=args.pnhid2, num_layers=1, batch_first=True)
        self.init_global = nn.Parameter(torch.FloatTensor(1, 1, args.pnhid2))
        self.init_global.data.uniform_(-1., 1.)
        print (self.init_global)

        self.Wq = nn.Linear(args.pnhid2, args.pnhid2, bias=False)
        self.Wk = nn.Linear(args.pnhid2, args.pnhid2, bias=False)
        #self.output_layer = nn.Linear(args.pnhid2, 1, bias=False)
        #self.output_layer_1 = nn.Linear(2 * args.pnhid2, 2 * args.pnhid2, bias=True)
        #self.output_layer_2 = nn.Linear(2 * args.pnhid2, 2 * args.pnhid2, bias=True)
        #self.output_layer_3 = nn.Linear(2 * args.pnhid2, 1, bias=False)

    def forward(self, state, adj, selection_history=None):

        x = F.relu(self.gc1(state.transpose(0, 1), adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        x = F.relu(x)
        x = x.transpose(0, 1)

        if (selection_history.shape[1] != 0):
            #global_info = torch.stack([state[i, selection_history[i], :-1].mean(dim=0) for i in range(self.args.batchsize)], dim=0)
            global_info = torch.stack([state[i, selection_history[i], :-1] for i in range(self.args.batchsize)], dim=0)
            global_info, _ = self.global_rnn(global_info)
            global_info = global_info[:, -1, :].unsqueeze(1)
        else:
            global_info = self.init_global.expand([self.args.batchsize, -1, -1])
        global_info = global_info.expand([-1, x.size(1), -1])
        #global_info = F.relu(self.global_embed(global_info))
        
        logits = (self.Wk(x) * self.Wq(global_info)).sum(dim=2)
        #logits = self.output_layer(x * torch.sigmoid(global_info)).squeeze(-1)

        #x = torch.cat([x, global_info], dim=-1)
        #x = F.relu(self.output_layer_1(x))
        #x = F.relu(self.output_layer_2(x))
        #x = self.output_layer_3(x).squeeze(-1)

        if (selection_history.shape[1] == 0):
            print (torch.softmax(logits, dim=1).max(dim=1))
        
        return logits
'''
'''
# use graph encoding as a gating function on node features
class PolicyNet(nn.Module):

    def __init__(self,args,statedim):
        super(PolicyNet, self).__init__()
        self.args = args
        self.gc1 = GraphConvolution(statedim - args.use_degree_distribution, args.pnhid, args.batchsize, bias=True)
        self.gc2 = GraphConvolution(args.pnhid, args.pnhid2, args.batchsize, bias=True)
        self.lin = nn.Linear(args.pnhid2, 1, bias=True)
        self.dropout=nn.Dropout(p=args.pdropout, inplace=False)
        self.graph_embed = nn.Linear(args.use_degree_distribution, args.pnhid2, bias=True)

    def forward(self, state, adj):
        graph_summary = state[:, :, -self.args.use_degree_distribution:]
        state = state[:, :, :-self.args.use_degree_distribution] 

        x = F.relu(self.gc1(state.transpose(0,1), adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        x = F.relu(x)

        graph_summary = torch.sigmoid(self.graph_embed(graph_summary))
        x = self.lin((x.transpose(0, 1) * graph_summary)).squeeze(-1)
        return x
'''
'''
# use graph mean as graph encoding for attention query
class PolicyNet(nn.Module):

    def __init__(self,args,statedim):
        super(PolicyNet, self).__init__()
        self.args = args
        self.gc1 = GraphConvolution(statedim, args.pnhid, args.batchsize, bias=True)
        self.gc2 = GraphConvolution(args.pnhid, args.pnhid2, args.batchsize, bias=True)
        self.dropout=nn.Dropout(p=args.pdropout, inplace=False)
        self.graph_embed_1 = nn.Linear(args.pnhid2, 2 * args.pnhid2, bias=True)
        self.graph_embed_2 = nn.Linear(2 * args.pnhid2, args.pnhid2, bias=True)

        self.Wq = nn.Linear(args.pnhid2, args.pnhid2, bias=False)
        self.Wk = nn.Linear(args.pnhid2, args.pnhid2, bias=False)

    def forward(self, state, adj):

        #if (self.args.use_degree_distribution != 0):
        #    graph_summary = state[:, :, -self.args.use_degree_distribution:]
        #    state = state[:, :, :-self.args.use_degree_distribution]

        x = F.relu(self.gc1(state.transpose(0,1), adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        x = F.relu(x)

        graph_summary = x.mean(dim=0, keepdim=True).expand(x.size())
        graph_summary = F.relu(self.graph_embed_1(graph_summary))
        graph_summary = self.graph_embed_2(graph_summary)
        
        x = (self.Wk(x) * self.Wq(graph_summary)).sum(dim=2).transpose(0, 1)
        return x
'''