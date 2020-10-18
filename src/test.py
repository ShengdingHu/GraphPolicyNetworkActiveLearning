from src.utils.player import Player
from src.utils.query import RandomQuery
from src.utils.dataloader import GraphLoader
import random
import numpy as np
import argparse
import time
from functools import reduce
from src.utils.common import logger
from src.utils.utils import *
from src.utils.query import ProbQuery
from src.utils.policynet import *
import scipy.stats as stats


switcher = {'gcn':PolicyNet,'mlp':PolicyNet2}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nhid",type=int,default=64)
    parser.add_argument("--pnhid",type=str,default='8+8')
    parser.add_argument("--pnout",type=int,default=1)
    parser.add_argument("--pdropout",type=float,default=0.0)
    parser.add_argument("--dropout",type=float,default=0.2)
    parser.add_argument("--lr",type=float,default=3e-2)
    parser.add_argument("--batchsize",type=int,default=10,help="here batchsize means the number of "
                                                               "repeated (independence) experiment of testing")
    parser.add_argument("--budgets",type=str, help="budget for label query")
    parser.add_argument("--ntest",type=int,default=1000)
    parser.add_argument("--nval",type=int,default=500)
    parser.add_argument("--datasets",type=str,default="cora")
    parser.add_argument("--modelname",type=str,default="tmp")
    parser.add_argument("--remain_epoch",type=int,default=35,help="continues training $remain_epoch"
                                                                  " epochs after all the selection")
    parser.add_argument("--method",type=str,default='1',help="1 for random,2 for age,3 for policy,4 for entropy, 5 for centrality")

    parser.add_argument("--experimentnum",type=int,default=100)
    parser.add_argument("--fix_test",type=int,default=0)
    parser.add_argument("--multigraphindex", type=str, default="")
    
    parser.add_argument("--use_entropy",type=int,default=1)
    parser.add_argument("--use_degree",type=int,default=1)
    parser.add_argument("--use_local_diversity",type=int,default=1)
    parser.add_argument("--use_select",type=int,default=1)
    
    parser.add_argument("--age_basef", type=float, default=0.95)

    parser.add_argument("--same_initialization", type=int, default=0)
    parser.add_argument("--policynet",type=str,default='gcn')

    args = parser.parse_args()
    args.pnhid = [int(n) for n in args.pnhid.split('+')]
    return args


def randomQuery(p, args):

    q = RandomQuery()
    G = p.G
    for i in range(args.current_budget):
        pool = p.getPool()
        selected = q(pool)
        p.query(selected)
        p.trainOnce(log=True)

    for i in range(args.remain_epoch):
        p.trainOnce()

    acc_val = p.validation()
    acc_test = p.validation(test=True)
    return acc_test


def ageQuery(p, args):

    from src.baselines.age import AGEQuery
    G = p.G
    q = AGEQuery(G,args.age_basef)

    for i in range(args.current_budget):
        pool = p.getPool()
        output = torch.nn.functional.softmax(p.allnodes_output.transpose(1,2),dim=2).detach().cpu().numpy()
        selected = q(output,pool,i)
        logger.debug("selected nodes {}".format(selected))
        p.query(selected)
        p.trainOnce(log=True)

    for i in range(args.remain_epoch):
        p.trainOnce(log=True)

    acc_val = p.validation()
    acc_test = p.validation(test=True)
    return acc_test


def entropyQuery(p, args):

    from src.baselines.age import EntropyQuery
    G = p.G
    q = EntropyQuery(G,args.age_basef)

    for i in range(args.current_budget):
        pool = p.getPool()
        output = torch.nn.functional.softmax(p.allnodes_output.transpose(1,2),dim=2).detach().cpu().numpy()
        selected = q(output,pool,i)
        logger.debug("selected nodes {}".format(selected))
        p.query(selected)
        p.trainOnce(log=True)

    for i in range(args.remain_epoch):
        p.trainOnce(log=True)

    acc_val = p.validation()
    acc_test = p.validation(test=True)
    return acc_test


def centralityQuery(p, args):

    from src.baselines.age import CentralityQuery
    G = p.G
    q = CentralityQuery(G,args.age_basef)

    for i in range(args.current_budget):
        pool = p.getPool()
        output = torch.nn.functional.softmax(p.allnodes_output.transpose(1,2),dim=2).detach().cpu().numpy()
        selected = q(output,pool,i)
        logger.debug("selected nodes {}".format(selected))
        p.query(selected)
        p.trainOnce(log=True)

    for i in range(args.remain_epoch):
        p.trainOnce(log=True)

    acc_val = p.validation()
    acc_test = p.validation(test=True)
    return acc_test


def coresetQuery(p, args):

    from src.baselines.coreset import CoreSetQuery
    G = p.G
    notrainmask = p.testmask + p.trainmask
    row,col = torch.where(notrainmask<0.1)
    row, col = row.cpu().numpy(),col.cpu().numpy()
    trainsetid = []
    for i in range(args.batchsize):
        trainsetid.append(col[row==i].tolist())

    q = CoreSetQuery(args.batchsize,trainsetid)

    for i in range(args.current_budget):
        pool = p.getPool()
        output = torch.nn.functional.softmax(p.allnodes_output.transpose(1, 2), dim=2).detach().cpu().numpy()
        validoutput = output[row, col].reshape((args.batchsize,len(trainsetid[0]),-1))
        selected = q(validoutput)
        logger.debug("selected nodes {}".format(selected))
        p.query(selected)
        p.trainOnce(log=False)

    for i in range(args.remain_epoch):
        p.trainOnce(log=False)

    acc_val = p.validation()
    acc_test = p.validation(test=True)
    return acc_test


def anrmabQuery(p, args):

    from src.baselines.anrmab import AnrmabQuery
    G = p.G
    q = AnrmabQuery(G, args.current_budget,G.stat["nnode"]-args.ntest-args.nval,args.batchsize)

    lastselect = None
    for i in range(args.current_budget):
        pool = p.getPool()
        output = torch.nn.functional.softmax(p.allnodes_output.transpose(1, 2), dim=2).detach().cpu().numpy()

        if lastselect is not None:
            lastselectedoutput = output[range(len(lastselect)), lastselect]
            lastpred = np.argmax(lastselectedoutput,axis=-1)
            truelabel = G.Y[lastselect].cpu().numpy()
            lastselectacc = (truelabel == lastpred).astype(np.float)
        else:
            lastselectacc = [0. for i in range(args.batchsize)]
        selected = q(output, lastselectacc, pool)
        lastselect = selected
        logger.debug("selected nodes {}".format(selected))
        p.query(selected)
        p.trainOnce(log=False)

    for i in range(args.remain_epoch):
        p.trainOnce(log=False)

    acc_val = p.validation()
    acc_test = p.validation(test=True)
    return acc_test


def policyQuery(p, args, policy=None):

    from src.utils.env import Env
    from src.utils.query import ProbQuery
    G = p.G
    e = Env([p],args)
    if policy is None:
        policy = switcher[args.policynet](args,e.statedim).cuda()
        policy.load_state_dict(torch.load("models/{}.pkl".format(args.modelname)))
    
    q = ProbQuery("hard")

    action_index = np.zeros([args.batchsize, args.current_budget])
    for i in range(args.current_budget):
        pool = p.getPool(reduce=False)
        state = e.getState()
        logits = policy(state,G.normadj)
        selected = q(logits,pool)
        action_index[:, i] = selected.detach().cpu().numpy()
        logger.debug("selected nodes {}".format(selected))
        p.query(selected)
        p.trainOnce(log=False)

    for i in range(args.remain_epoch):
        p.trainOnce(log=False)

    acc_val = p.validation()
    acc_test = p.validation(test=True)
    return acc_test


def multipleRun(p, args, times=100, budget=None):

    method = {"1":randomQuery,"2":ageQuery,"3":policyQuery,"4":entropyQuery,"5":centralityQuery,"6":coresetQuery,"7":anrmabQuery}
    ave = []

    for time in range(times):
        p.reset(fix_test=args.fix_test)
        acc_test = method[args.method](p, args)
        ave.append(acc_test)
        if (time % 10 == 0):
            print (time)
    ave = list(zip(*ave))
    ave = [reduce(lambda x,y:x+y, i) for i in ave]
    stat_ave = [mean_std(L) for L in ave]
    print(stat_ave)


'''
def sameIntializationTest(p,args,times=100):
    method = {"1":randomQuery,"2":ageQuery,"3":policyQuery,"4":entropyQuery,"5":centralityQuery}
    ave1, ave2, ave3 = [], [], []
    ave1macro,ave2macro,ave3macro = [],[],[]
    for time in range(times):
        p.reset(fix_test=args.fix_test)
        torch.save(p.net.state_dict(),"tmpfile_net_params.pkl")
        
        acc_test1, _ = method["1"](p, args,test_no_seq=False)
        ave1.extend(list(acc_test1[0]))
        ave1macro.extend(list(acc_test1[1]))

        p.reset(resplit=False)
        p.net.load_state_dict(torch.load("tmpfile_net_params.pkl"))
        acc_test2, _ = method["2"](p, args,test_no_seq = False)
        ave2.extend(list(acc_test2[0]))
        ave2macro.extend(list(acc_test2[1]))

        p.reset(resplit=False)
        p.net.load_state_dict(torch.load("tmpfile_net_params.pkl"))
        acc_test3, _ = method["3"](p, args,test_no_seq = False)
        ave3.extend(list(acc_test3[0]))
        ave3macro.extend(list(acc_test3[1]))
    
    ave1, ave1macro = np.array(ave1), np.array(ave1macro)
    ave2, ave2macro = np.array(ave2), np.array(ave2macro)
    ave3, ave3macro = np.array(ave3), np.array(ave3macro)

    improveby2 = ave2 - ave1
    improveby3 = ave3 - ave1
    
    ratio = improveby3 - improveby2
    
    averatio,stdratio = mean_std(ratio)
    ttestresult = stats.ttest_rel(improveby3,improveby2)
    print("mean difference between AGE and ours {}, std {}".format(averatio,stdratio))
    print("paired ttest p value {}".format(ttestresult.pvalue))    
'''



if __name__=="__main__":

    args = parse_args()
    datasets = args.datasets.split('+')
    budgets = [int(x) for x in args.budgets.split('+')]
    if (datasets[0] == "reddit1401"):
        multigraphindex = args.multigraphindex.split("+")
        for i in range(len(multigraphindex)):
            print (datasets[0], multigraphindex[i])
            args.current_budget = budgets[i]
            print('budgets: %d' % (args.current_budget))
            G = GraphLoader(datasets[0], sparse=True, multigraphindex=multigraphindex[i], args=args)
            G.process()
            p = Player(G, args)
            multipleRun(p, args, times=args.experimentnum)
    else:
        for i in range(len(datasets)):
            print (datasets[i])
            args.current_budget = budgets[i]
            print('budgets: %d' % (args.current_budget))
            G = GraphLoader(datasets[i], sparse=True, multigraphindex=args.multigraphindex, args=args)
            G.process()
            p = Player(G, args)
            multipleRun(p, args, times=args.experimentnum)