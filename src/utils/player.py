# individual player who takes the action and evaluates the effect
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np

from src.utils.common import *
from src.utils.classificationnet import GCN
from src.utils.utils import *


class Player(nn.Module):

    def __init__(self,G,args,rank=0):

        super(Player,self).__init__()
        self.G = G
        self.args = args
        self.rank = rank
        self.batchsize = args.batchsize

        if self.G.stat['multilabel']:
            self.net = GCN(self.G.stat['nfeat'],args.nhid,self.G.stat['nclass'],args.batchsize,args.dropout,False,bias=True).cuda()
            self.loss_func=F.binary_cross_entropy_with_logits
        else:
            self.net = GCN(self.G.stat['nfeat'],args.nhid,self.G.stat['nclass'],args.batchsize,args.dropout,True).cuda()
            self.loss_func=F.nll_loss

        self.fulllabel = self.G.Y.expand([self.batchsize]+list(self.G.Y.size()))

        self.reset(fix_test=False) #initialize
        self.count = 0


    def makeValTestMask(self, fix_test=True):
        #if fix_test:
        #    assert False
        valmask=torch.zeros((self.batchsize,self.G.stat['nnode'])).to(torch.float).cuda()
        testmask = torch.zeros((self.batchsize,self.G.stat['nnode'])).to(torch.float).cuda()
        valid = []
        testid = []
        vallabel = []
        testlabel = []
        for i in range(self.batchsize):
            base = np.array([x for x in range(self.G.stat["nnode"])])
            if fix_test:
                testid_=[x for x in range(self.G.stat["nnode"] - self.args.ntest,self.G.stat["nnode"])]
            else:
                testid_ = np.sort(np.random.choice(base, size=self.args.ntest, replace=False)).tolist()
            testmask[i, testid_] = 1.
            testid.append(testid_)
            testlabel.append(self.G.Y[testid_])
            s = set(testid_)
            base= [x for x in range(self.G.stat["nnode"]) if x not in s ]
            valid_ = np.sort(np.random.choice(base, size=self.args.nval, replace=False)).tolist()
            valmask[i,valid_]=1.
            valid.append(valid_)
            vallabel.append(self.G.Y[valid_])
        self.valid = torch.tensor(valid).cuda()
        self.testid = torch.tensor(testid).cuda()
        self.vallabel = torch.stack(vallabel).cuda()
        self.testlabel = torch.stack(testlabel).cuda()
        self.valmask=valmask
        self.testmask=testmask


    def lossWeighting(self,epoch):
        return min(epoch,10.)/10.


    def query(self,nodes):
        self.trainmask[[x for x in range(self.batchsize)],nodes] = 1.


    def getPool(self,reduce=True):
        mask = self.testmask+self.valmask+self.trainmask
        row,col = torch.where(mask<0.1)
        if reduce:
            row, col = row.cpu().numpy(),col.cpu().numpy()
            pool = []
            for i in range(self.batchsize):
                pool.append(col[row==i])
            return pool
        else:
            return row,col


    def trainOnce(self,log=False):
        nlabeled = torch.sum(self.trainmask)/self.batchsize
        self.net.train()
        self.opt.zero_grad()
        output = self.net(self.G.X,self.G.normadj)
        # print(output.size())
        # exit()
        if self.G.stat["multilabel"]:
            output_trans = output.transpose(1,2)
            # print(output_trans[:,-20:,:])
            
            losses = self.loss_func(output_trans,self.fulllabel,reduction="none").sum(dim=2)
        else:
            losses = self.loss_func(output,self.fulllabel,reduction="none")
        loss = torch.sum(losses*self.trainmask)/nlabeled*self.lossWeighting(float(nlabeled.cpu()))
        loss.backward()
        self.opt.step()
        #if log:
            #logger.info("nnodes selected:{},loss:{}".format(nlabeled,loss.detach().cpu().numpy()))
        self.allnodes_output=output.detach()
        return output


    def validation(self,test=False,rerun=True):
        if test:
            mask = self.testmask
            labels= self.testlabel
            index = self.testid
        else:
            mask = self.valmask
            labels = self.vallabel
            index = self.valid
        if rerun:
            self.net.eval()
            output = self.net(self.G.X,self.G.normadj)
        else:
            output = self.allnodes_output
        if self.G.stat["multilabel"]:
            # logger.info("output of classification {}".format(output))
            output_trans = output.transpose(1,2)
            losses_val = self.loss_func(output_trans,self.fulllabel,reduction="none").mean(dim=2)
        else:
            losses_val = self.loss_func(output,self.fulllabel,reduction="none")
        loss_val = torch.sum(losses_val*mask,dim =1,keepdim=True)/torch.sum(mask,dim =1,keepdim=True)
        acc= []
        for i in range(self.batchsize):
            pred_val = (output[i][:,index[i]]).transpose(0,1)
            # logger.info("pred_val {}".format(pred_val))
            acc.append(accuracy(pred_val,labels[i]))

        # logger.info("validation acc {}".format(acc))
        return list(zip(*acc))


    def trainRemain(self):
        for i in range(self.args.remain_epoch):
            self.trainOnce()


    def reset(self,resplit=True,fix_test=True):
        if resplit:
            self.makeValTestMask(fix_test=fix_test)
        self.trainmask = torch.zeros((self.batchsize,self.G.stat['nnode'])).to(torch.float).cuda()
        self.net.reset()
        self.opt = torch.optim.Adam(self.net.parameters(),lr=self.args.lr,weight_decay=5e-4)
        self.allnodes_output = self.net(self.G.X,self.G.normadj).detach()



import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout",type=float,default=0.5)
    parser.add_argument("--ntest",type=int,default=1000)
    parser.add_argument("--nval",type=int,default=500)
    parser.add_argument("--nhid", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--budget", type=int, default=20, help="budget per class")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--remain_epoch", type=int, default=35, help="continues training $remain_epoch")

    args = parser.parse_args()
    return args

if __name__=="__main__":
    from src.utils.dataloader import GraphLoader
    args = parse_args()
    G = GraphLoader("cora")
    G.process()
    p = Player(G,args)
    p.query([2,3])
    p.query([4,6])

    p.trainOnce()

    print(p.trainmask[:,:10])
    print(p.allnodes_output[0].size())