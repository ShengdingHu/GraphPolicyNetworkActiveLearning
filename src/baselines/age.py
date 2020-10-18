## this is the code borrowed from AGE's public code
import time
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import scipy as sc
import networkx as nx
import torch

def centralissimo(G):
    centralities = []
    centralities.append(nx.pagerank(G))  #centralities.append(nx.harmonic_centrality(G))
    L = len(centralities[0])
    Nc = len(centralities)
    cenarray = np.zeros((Nc,L))
    for i in range(Nc):
        cenarray[i][list(centralities[i].keys())]=list(centralities[i].values())
    normcen = (cenarray.astype(float)-np.min(cenarray,axis=1)[:,None])/(np.max(cenarray,axis=1)-np.min(cenarray,axis=1))[:,None]
    return normcen

def multiclassentropy_numpy(tens,dim=1):
    reverse = 1 - tens
    ent_1 = -np.log(np.clip(tens, a_min=1e-7,a_max=None)) * tens
    ent_2 = -np.log(np.clip(reverse, a_min=1e-7,a_max=None)) * reverse
    ent = ent_1 + ent_2
    entropy = np.mean(ent, axis=1)

    return entropy

class AGEQuery(object):
    def __init__(self,G,basef=0.95):
        self.G = G
        self.normcen = centralissimo(self.G.G)[0]
        self.cenperc = self.perc(self.normcen)
        self.basef = basef
        print("age basef is {}".format(self.basef))
        self.multilabel=self.G.stat["multilabel"]
        self.NCL = self.G.stat["nclass"]

    def __call__(self,outputs,pool,epoch):
        ret = []
        for id,row in enumerate(pool):
            selected = self.selectOneNode(outputs[id],row,epoch)
            ret.append(selected)
        ret = np.array(ret)#.reshape(-1,1)
        return ret


    def selectOneNode(self,output,pool,epoch):
        gamma = np.random.beta(1, 1.005 - self.basef ** epoch)
        alpha = beta = (1 - gamma) / 2

        if self.multilabel:
            probs = 1. / (1. + np.exp(-output))
            entropy = multiclassentropy_numpy(probs)
        else:
            entropy = sc.stats.entropy(output.transpose())
        assert type(entropy) == np.ndarray, "entropy type {}".format(type(entropy))

        entrperc = self.perc(entropy)
        kmeans = KMeans(n_clusters=self.NCL, random_state=0).fit(output)
        ed = euclidean_distances(output, kmeans.cluster_centers_)
        ed_score = np.min(ed,axis=1)  # the larger ed_score is, the far that node is away from cluster
                                      # centers, the less representativeness the node is
        edprec = self.percd(ed_score)
        finalweight = alpha * entrperc + beta * edprec + gamma * self.cenperc
        finalweight = finalweight[pool]
        select = pool[np.argmax(finalweight)]
        return select

    # calculate the percentage of elements larger than the k-th element
    def percd(self,input):
        return 1-np.argsort(np.argsort(input,kind='stable'),kind='stable')/len(input)

    # calculate the percentage of elements smaller than the k-th element
    def perc(self,input):
        return 1-np.argsort(np.argsort(-input,kind='stable'),kind='stable')/len(input)

class EntropyQuery(object):
    def __init__(self,G,basef=0.95):
        self.G = G
        self.normcen = centralissimo(self.G.G)[0]
        self.cenperc = self.perc(self.normcen)
        self.basef = basef
        print("age basef is {}".format(self.basef))
        self.multilabel=self.G.stat["multilabel"]
        self.NCL = self.G.stat["nclass"]

    def __call__(self,outputs,pool,epoch):
        ret = []
        for id,row in enumerate(pool):
            selected = self.selectOneNode(outputs[id],row,epoch)
            ret.append(selected)
        ret = np.array(ret)#.reshape(-1,1)
        return ret


    def selectOneNode(self,output,pool,epoch):
        gamma = np.random.beta(1, 1.005 - self.basef ** epoch)
        alpha = beta = (1 - gamma) / 2

        if self.multilabel:
            probs = 1. / (1. + np.exp(-output))
            entropy = multiclassentropy_numpy(probs)
        else:
            entropy = sc.stats.entropy(output.transpose())
        assert type(entropy) == np.ndarray, "entropy type {}".format(type(entropy))

        entrperc = self.perc(entropy)
        # kmeans = KMeans(n_clusters=self.NCL, random_state=0).fit(output)
        # ed = euclidean_distances(output, kmeans.cluster_centers_)
        # ed_score = np.min(ed,axis=1)  # the larger ed_score is, the far that node is away from cluster
        #                               # centers, the less representativeness the node is
        # edprec = self.percd(ed_score)
        finalweight = entrperc #+ beta * edprec + gamma * self.cenperc
        finalweight = finalweight[pool]
        select = pool[np.argmax(finalweight)]
        return select

    # calculate the percentage of elements larger than the k-th element
    def percd(self,input):
        return 1-np.argsort(np.argsort(input,kind='stable'),kind='stable')/len(input)

    # calculate the percentage of elements smaller than the k-th element
    def perc(self,input):
        return 1-np.argsort(np.argsort(-input,kind='stable'),kind='stable')/len(input)


class CentralityQuery(object):
    def __init__(self,G,basef=0.95):
        self.G = G
        self.normcen = centralissimo(self.G.G)[0]
        self.cenperc = self.perc(self.normcen)
        self.basef = basef
        print("age basef is {}".format(self.basef))
        self.multilabel=self.G.stat["multilabel"]
        self.NCL = self.G.stat["nclass"]

    def __call__(self,outputs,pool,epoch):
        ret = []
        for id,row in enumerate(pool):
            selected = self.selectOneNode(outputs[id],row,epoch)
            ret.append(selected)
        ret = np.array(ret)#.reshape(-1,1)
        return ret


    def selectOneNode(self,output,pool,epoch):
        gamma = np.random.beta(1, 1.005 - self.basef ** epoch)
        alpha = beta = (1 - gamma) / 2

        # if self.multilabel:
        #     probs = 1. / (1. + np.exp(-output))
        #     entropy = multiclassentropy_numpy(probs)
        # else:
        #     entropy = sc.stats.entropy(output.transpose())
        # assert type(entropy) == np.ndarray, "entropy type {}".format(type(entropy))

        # entrperc = self.perc(entropy)
        # kmeans = KMeans(n_clusters=self.NCL, random_state=0).fit(output)
        # ed = euclidean_distances(output, kmeans.cluster_centers_)
        # ed_score = np.min(ed,axis=1)  # the larger ed_score is, the far that node is away from cluster
        #                               # centers, the less representativeness the node is
        # edprec = self.percd(ed_score)
        finalweight = self.cenperc
        finalweight = finalweight[pool]
        select = pool[np.argmax(finalweight)]
        return select

    # calculate the percentage of elements larger than the k-th element
    def percd(self,input):
        return 1-np.argsort(np.argsort(input,kind='stable'),kind='stable')/len(input)

    # calculate the percentage of elements smaller than the k-th element
    def perc(self,input):
        return 1-np.argsort(np.argsort(-input,kind='stable'),kind='stable')/len(input)


class EdgeQuery(object):
    def __init__(self,G):
        self.G = G
        self.adj = self.G.adj.cpu().to_dense().numpy()
        self.adjidx = np.where(self.adj>0)
        self.adj += np.eye(self.adj.shape[0])
        self.normcen = centralissimo(self.G.G)[0]
        self.cenperc = self.perc(self.normcen)
        self.basef = 0.95
        self.multilabel=self.G.stat["multilabel"]
        self.NCL = self.G.stat["nclass"]

    def __call__(self,outputs,pool,epoch):
        ret = []
        for id,row in enumerate(pool):
            selected = self.selectOneNode(outputs[id],row,epoch)
            ret.append(selected)
        ret = np.array(ret)#.reshape(-1,1)
        return ret


    def selectOneNode(self,output,pool,epoch):
        gamma = np.random.beta(1, 1.005 - self.basef ** epoch)
        alpha = beta = (1 - gamma) / 2



        if self.multilabel:
            probs = 1. / (1. + np.exp(-output))
            entropy = multiclassentropy_numpy(probs)
        else:
            entropy = sc.stats.entropy(output.transpose())
        assert type(entropy) == np.ndarray, "entropy type {}".format(type(entropy))

        entropy/=np.log(float(self.G.stat['nclass']))
        row,col = self.adjidx


        N=entropy.shape[0]
        t0= time.time()

        b=entropy[row]
        c=entropy[col]
        d = np.vstack([b,c])

        weight = np.array([0.8,0.2]).transpose()

        
       
        e = np.matmul(weight,d)
        eta = 1.5
        e = eta*(e-0.5)+0.5

        f = sp.csr_matrix((e,(row,col)),shape=(N,N))

        g = np.asarray( np.sum(f,axis=1))
        g = np.squeeze(g,axis = 1)
        # entrperc = self.perc(entropy)
        # kmeans = KMeans(n_clusters=self.NCL, random_state=0).fit(output)
        # ed = euclidean_distances(output, kmeans.cluster_centers_)
        # ed_score = np.min(ed,axis=1)  # the larger ed_score is, the far that node is away from cluster
        #                               # centers, the less representativeness the node is
        finalweight = self.perc(g)
        # finalweight = alpha * entrperc + beta * edprec + gamma * self.cenperc
        finalweight = finalweight[pool]
        select = pool[np.argmax(finalweight)]
        return select

    # calculate the percentage of elements larger than the k-th element
    def percd(self,input):
        return 1-np.argsort(np.argsort(input,kind='stable'),kind='stable')/len(input)

    # calculate the percentage of elements smaller than the k-th element
    def perc(self,input):
        return 1-np.argsort(np.argsort(-input,kind='stable'),kind='stable')/len(input)


