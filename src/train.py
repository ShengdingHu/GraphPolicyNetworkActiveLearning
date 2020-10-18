import numpy as np
import torch
import argparse
from pprint import pformat
from torch.distributions import Categorical
import torch.nn.functional as F
from src.utils.dataloader import GraphLoader
from src.utils.player import Player
from src.utils.env import Env
from src.utils.policynet import *
from src.utils.rewardshaper import RewardShaper
from src.utils.common import *
from src.utils.utils import *
from src.utils.const import MIN_EPSILON


switcher = {'gcn':PolicyNet,'mlp':PolicyNet2}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nhid",type=int,default=64)
    parser.add_argument("--pnhid",type=str,default='8+8')
    parser.add_argument("--dropout",type=float,default=0.2)
    parser.add_argument("--pdropout",type=float,default=0.0)
    parser.add_argument("--lr",type=float,default=3e-2)
    parser.add_argument("--rllr",type=float,default=1e-2)
    parser.add_argument("--entcoef",type=float,default=0)
    parser.add_argument("--frweight",type=float,default=1e-3)
    parser.add_argument("--batchsize",type=int,default=10)
    parser.add_argument("--budgets",type=str,default="35",help="budget per class")
    parser.add_argument("--ntest",type=int,default=1000)
    parser.add_argument("--nval",type=int,default=500)
    parser.add_argument("--datasets",type=str,default="cora")
    parser.add_argument("--metric",type=str,default="microf1")
    parser.add_argument("--remain_epoch",type=int,default=35,help="continues training $remain_epoch"
                                                                  " epochs after all the selection")
    parser.add_argument("--shaping",type=str,default="234",help="reward shaping method, 0 for no shaping;"
                                                              "1 for add future reward,i.e. R= r+R*gamma;"
                                                              "2 for use finalreward;"
                                                              "3 for subtract baseline(value of curent state)"
                                                              "1234 means all the method is used,")
    parser.add_argument("--logfreq",type=int,default=10)
    parser.add_argument("--maxepisode",type=int,default=20000)
    parser.add_argument("--save",type=int,default=0)
    parser.add_argument("--savename",type=str,default="tmp")
    parser.add_argument("--policynet",type=str,default='gcn')
    parser.add_argument("--multigraphindex", type=int, default=0)

    parser.add_argument("--use_entropy",type=int,default=1)
    parser.add_argument("--use_degree",type=int,default=1)
    parser.add_argument("--use_local_diversity",type=int,default=1)
    parser.add_argument("--use_select",type=int,default=1)

    parser.add_argument('--pg', type=str, default='reinforce')
    parser.add_argument('--ppo_epoch', type=int, default=5)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--schedule', type=int, default=0)

    args = parser.parse_args()
    logargs(args,tablename="config")
    args.pnhid = [int(n) for n in args.pnhid.split('+')]
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    return args


class SingleTrain(object):

    def __init__(self, args):
    
        self.globel_number = 1
        self.args = args
        self.datasets = self.args.datasets.split("+")
        self.budgets = [int(x) for x in self.args.budgets.split("+")]
        self.graphs, self.players, self.rshapers, self.accmeters = [], [], [], []
        for i, dataset in enumerate(self.datasets):
            g = GraphLoader(dataset, sparse=True,args=args, multigraphindex='graph' + str(i + 1))
            g.process()
            self.graphs.append(g)
            p = Player(g, args).cuda()
            self.players.append(p)
            self.rshapers.append(RewardShaper(args))
            self.accmeters.append(AverageMeter("accmeter",ave_step=100))
        self.env = Env(self.players,args)

        self.policy=switcher[args.policynet](self.args,self.env.statedim).cuda()

        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.args.rllr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [1000, 3000], gamma=0.1, last_epoch=-1)
        self.accmeter = (AverageMeter("aveaccmeter",ave_step=100))


    def jointtrain(self, maxepisode):

        for episode in range(1,maxepisode):
            for playerid in range(len(self.datasets)):
                shapedrewards, logp_actions, p_actions = self.playOneEpisode(episode, playerid=playerid)
                if episode > 10:
                    loss = self.finishEpisode(shapedrewards, logp_actions, p_actions)
                else:
                    loss = None
                #if episode%self.args.logfreq == 0:
                    #logger.info("episode {}, playerid {}, loss {}".format(episode,playerid,loss))
                    #inspect_weight(self.policy)
                if self.args.save==1 and self.accmeter.should_save():
                    logger.warning("saving!")
                    torch.save(self.policy.state_dict(),"models/{}.pkl".format(self.args.policynet+self.args.savename))
            if (episode % 100 == 1):
                torch.save(self.policy.state_dict(),"models/{}.pkl".format(self.args.policynet+self.args.savename+'_'+str(episode)))                
            self.globel_number +=1

        
    def playOneEpisode(self, episode, playerid=0):

        self.playerid = playerid
        self.env.reset(playerid)
        rewards, logp_actions, p_actions = [], [], []
        self.states, self.actions, self.pools = [], [], []
        initialrewards=self.env.players[playerid].validation()
        rewards.append(initialrewards)
        self.entropy_reg = []
        self.action_index = np.zeros([self.args.batchsize, self.budgets[playerid]])
        for epoch in range(self.budgets[playerid]):
            state = self.env.getState(playerid)
            self.states.append(state)
            pool = self.env.players[playerid].getPool(reduce=False)
            self.pools.append(pool)

            logits = self.policy(state,self.graphs[playerid].normadj)
            action,logp_action, p_action = self.selectActions(logits,pool)
            self.action_index[:, epoch] = action.detach().cpu().numpy()
            logp_actions.append(logp_action)
            p_actions.append(p_action)
            rewards.append(self.env.step(action,playerid))

            self.entropy_reg.append(-(self.valid_probs * torch.log(1e-6 + self.valid_probs)).sum(dim=1) / np.log(self.valid_probs.size(1)))
            ##
            # if episode % self.args.logfreq == 0:
            #     logger.debug("probs {}".format(logits[0].max()))
            #     logger.debug("action's state{} ".format(state[0,action[0],:]))
        #if episode % self.args.logfreq == 0:
            #print ('normalized entropy of the current policy')
            #print (self.entropy_reg.mean(dim=1) / np.log(logits.size(1)))
        
        self.env.players[playerid].trainRemain()
        logp_actions = torch.stack(logp_actions)
        p_actions = torch.stack(p_actions)
        self.entropy_reg = torch.stack(self.entropy_reg).cuda()
        finalrewards = self.env.players[playerid].validation(rerun=True)
        micfinal, _ = mean_std(finalrewards[0])
        self.accmeters[playerid].update(micfinal)
        self.accmeter.update(micfinal)
        if episode % self.args.logfreq == 0:
            logger.info("episode {},playerid{}. acc in validation {},aveacc {}".format(episode, playerid, micfinal, self.accmeters[playerid]()))
        shapedrewards = self.rshapers[playerid].reshape(rewards,finalrewards,logp_actions.detach().cpu().numpy())
        return shapedrewards,logp_actions, p_actions


    def finishEpisode(self,rewards,logp_actions, p_actions):

        rewards = torch.from_numpy(rewards).cuda().type(torch.float32)
        
        if (self.args.pg == 'reinforce'):
            #losses =torch.sum(-logp_actions*rewards,dim=0)
            #loss = torch.mean(losses) - self.args.entcoef * self.entropy_reg.sum(dim=0).mean()
            losses = logp_actions * rewards + self.args.entcoef * self.entropy_reg
            loss = -torch.mean(torch.sum(losses, dim=0))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if self.args.schedule:
                self.scheduler.step()
        elif (self.args.pg == 'ppo'):
            epsilon = 0.2
            p_old = p_actions.detach()
            r_sign = torch.sign(rewards).type(torch.float32)
            for i in range(self.args.ppo_epoch):
                if (i != 0):
                    p_actions = [self.trackActionProb(self.states[i], self.pools[i], self.actions[i]) for i in range(len(self.states))]
                    p_actions = torch.stack(p_actions)
                ratio = p_actions / p_old
                losses = torch.min(ratio * rewards, (1 + epsilon * r_sign) * rewards)
                loss = -torch.mean(losses)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        return loss.item()


    def trackActionProb(self, state, pool, action):
        logits = self.policy(state, self.graphs[self.playerid].normadj)
        valid_logits = logits[pool].reshape(self.args.batchsize,-1)
        max_logits = torch.max(valid_logits,dim=1,keepdim=True)[0].detach()
        valid_logits = valid_logits - max_logits
        valid_probs = F.softmax(valid_logits, dim=1)
        prob = valid_probs[list(range(self.args.batchsize)), action]
        return prob


    def selectActions(self,logits,pool):
        valid_logits = logits[pool].reshape(self.args.batchsize,-1)
        max_logits = torch.max(valid_logits,dim=1,keepdim=True)[0].detach()
        valid_logits = valid_logits - max_logits
        valid_probs = F.softmax(valid_logits, dim=1)
        self.valid_probs = valid_probs
        pool = pool[1].reshape(self.args.batchsize,-1)
        assert pool.size()==valid_probs.size()

        m = Categorical(valid_probs)
        action_inpool = m.sample()
        self.actions.append(action_inpool)
        logprob = m.log_prob(action_inpool)
        prob = valid_probs[list(range(self.args.batchsize)), action_inpool]
        action = pool[[x for x in range(self.args.batchsize)],action_inpool]
    
        return action, logprob, prob

    

if __name__ == "__main__":

    args = parse_args()
    singletrain = SingleTrain(args)
    singletrain.jointtrain(args.maxepisode)   