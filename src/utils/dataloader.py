import os
import networkx as nx
import pickle as pkl
import torch
import numpy as np
from collections import OrderedDict
import time
from src.utils.common import *
from src.utils.utils import *


class GraphLoader(object):

    def __init__(self,name,root = "./data",undirected=True, hasX=True,hasY=True,header=True,sparse=True,multigraphindex=None,args=None):

        self.name = name
        self.undirected = undirected
        self.hasX = hasX
        self.hasY = hasY
        self.header = header
        self.sparse = sparse
        self.dirname = os.path.join(root,name)
        if name == "reddit1401":
            self.prefix = os.path.join(root, name,multigraphindex,multigraphindex)
        else:
            self.prefix = os.path.join(root,name,name)
        self._load()
        self._registerStat()
        self.printStat()


    def _loadConfig(self):
        file_name = os.path.join(self.dirname,"bestconfig.txt")
        f = open(file_name,'r')
        L = f.readlines()
        L = [x.strip().split() for x in L]
        self.bestconfig = {x[0]:x[1] for x in L if len(x)!=0}


    def _loadGraph(self, header = True):
        """
            load file in form:
            --------------------
            NUM_Of_NODE\n
            v1 v2\n
            v3 v4\n
            --------------------
        """
        file_name = self.prefix+".edgelist"
        if not header:
            logger.warning("You are reading an edgelist with no explicit number of nodes")
        if self.undirected:
            G = nx.Graph()
        else:
            G = nx.DiGraph()
        with open(file_name) as f:
            L = f.readlines()
            if header:
                num_node = int(L[0].strip())
                L = L[1:]
            edge_list = [[int(x) for x in e.strip().split()] for e in L]
            nodeset = set([x for e in edge_list for x in e])
            # if header:
                # assert min(nodeset) == 0 and max(nodeset) + 1 == num_node, "input standard violated {} ,{}".format(num_node,max(nodeset))
        
        if header:
            G.add_nodes_from([x for x in range(num_node)])
        else:
            G.add_nodes_from([x for x in range(max(nodeset)+1)])
        G.add_edges_from(edge_list)
        self.G = G

    def _loadX(self):
        self.X = pkl.load(open(self.prefix + ".x.pkl", 'rb'))
        self.X = self.X.astype(np.float32)
        if self.name in  ["coauthor_phy","corafull"]:
            self.X = self.X[:,:2000] # the coauthor_phy's feature is too large to fit in the memory.

    def _loadY(self):
        self.Y = pkl.load(open(self.prefix+".y.pkl",'rb'))#.astype(np.float32)

    def _getAdj(self):
        self.adj = nx.adjacency_matrix(self.G).astype(np.float32)

    def _toTensor(self,device=None):
        if device is None:
            if self.sparse:
                self.adj = sparse_mx_to_torch_sparse_tensor(self.adj).cuda()
                self.normadj = sparse_mx_to_torch_sparse_tensor(self.normadj).cuda()
            else:
                self.adj = torch.from_numpy(self.adj).cuda()
                self.normadj = torch.from_numpy(self.normadj).cuda()
            self.X = torch.from_numpy(self.X).cuda()
            self.Y = torch.from_numpy(self.Y).cuda()

    def _load(self):
        self._loadGraph(header=self.header)
        self._loadConfig()
        if self.hasX:
            self._loadX()
        if self.hasY:
            self._loadY()
        self._getAdj()

    def _registerStat(self):
        L=OrderedDict()
        L["name"] = self.name
        L["nnode"] = self.G.number_of_nodes()
        L["nedge"] = self.G.number_of_edges()
        L["nfeat"] = self.X.shape[1]
        L["nclass"] = self.Y.max() + 1
        L["sparse"] = self.sparse
        L["multilabel"] = False
        L.update(self.bestconfig)
        self.stat = L

    def process(self):
        if int(self.bestconfig['feature_normalize']):
            self.X = column_normalize(preprocess_features(self.X)) # take some time
        
        # self.X = self.X - self.X.min(axis=0)
        # print(np.where(self.X))
        # exit()
        # print(self.X[:3,:].tolist())
        self.normadj = preprocess_adj(self.adj)
        if not self.sparse:
            self.adj = self.adj.todense()
            self.normadj = self.normadj.todense()
        self._toTensor()
        
        self.normdeg = self._getNormDeg()


    def printStat(self):
        logdicts(self.stat,tablename="dataset stat")

    def _getNormDeg(self):
        self.deg = torch.sparse.sum(self.adj, dim=1).to_dense()
        normdeg =self.deg/ self.deg.max()
        return normdeg