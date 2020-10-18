# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bandit wrapper around base AL sampling methods.

Assumes adversarial multi-armed bandit setting where arms correspond to 
mixtures of different AL methods.

Uses EXP3 algorithm to decide which AL method to use to create the next batch.
Similar to Hsu & Lin 2015, Active Learning by Learning.
https://www.csie.ntu.edu.tw/~htlin/paper/doc/aaai15albl.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from src.baselines.sampling_methods.wrapper_sampler_def import AL_MAPPING, WrapperSamplingMethod


class BanditDiscreteSampler(WrapperSamplingMethod):
  """Wraps EXP3 around mixtures of indicated methods.

  Uses EXP3 mult-armed bandit algorithm to select sampler methods.
  """
  def __init__(self,budget,seed=123,
               reward_function = lambda AL_acc: AL_acc[-1],
               gamma=0.5,
               samplers=[{'methods':('margin','uniform'),'weights':(0,1)},
                         {'methods':('margin','uniform'),'weights':(1,0)}]):
    """Initializes sampler with indicated gamma and arms.

    Args:
      X: training data
      y: labels, may need to be input into base samplers
      seed: seed to use for random sampling
      reward_function: reward based on previously observed accuracies.  Assumes
        that the input is a sequence of observed accuracies.  Will ultimately be
        a class method and may need access to other class properties.
      gamma: weight on uniform mixture.  Arm probability updates are a weighted
        mixture of uniform and an exponentially weighted distribution.
        Lower gamma more aggressively updates based on observed rewards.
      samplers: list of dicts with two fields
        'samplers': list of named samplers
        'weights': percentage of batch to allocate to each sampler
    """

    self.name = 'bandit_discrete'
    np.random.seed(seed)
    self.seed = seed
    # self.initialize_samplers(samplers)
    self.samplers = samplers
    self.gamma = gamma
    self.n_arms = len(samplers)
    self.reward_function = reward_function

    self.pull_history = []
    self.acc_history = []
    self.w = np.ones(self.n_arms)
    self.x = np.zeros(self.n_arms)
    self.p = self.w / (1.0 * self.n_arms)
    self.probs = []
    self.selectionhistory = []
    self.num_arm = float(len(self.samplers))

    self.pmin = np.sqrt(np.ln(self.num_arm)/(self.num_arm*budget))

  def update_vars_arnmab(self, arm_pulled):
    reward = self.reward_function(self.acc_history)
    Qkstar = [self.samplers[arm_pulled][self.selectionhistory[-1]]
    phistar = sum([self.p[i] * self.samplers[i].valuelist[self.selectionhistory[-1]] for i in range(len(self.samplers)))
    rhat = Qkstar / phistar
    self.w = self.w*np.exp(self.pmin/2.0*(rhat))


  
  def update_vars(self, arm_pulled):
    reward = self.reward_function(self.acc_history)

    self.x = np.zeros(self.n_arms)
    self.x[arm_pulled] = reward / self.p[arm_pulled]
    self.w = self.w * np.exp(self.gamma * self.x / self.n_arms)
    self.p = ((1.0 - self.gamma) * self.w / sum(self.w)
              + self.gamma / self.n_arms)
    # print(self.p)
    self.probs.append(self.p)

  def select_batch_arnmab(N, eval_acc, wraped_feat) :
    self.acc_history.append(eval_acc)

    
    if len(self.pull_history) > 0:
      self.update_vars(self.pull_history[-1])
    
  def select_batch(self, N, eval_acc,wraped_feat):
    """Returns batch of datapoints sampled using mixture of AL_methods.

    Assumes that data has already been shuffled.

    Args:
      already_selected: index of datapoints already selected
      N: batch size
      eval_acc: accuracy of model trained after incorporating datapoints from
        last recommended batch

    Returns:
      indices of points selected to label
    """

    # print("eval_acc {}".format(eval_acc))
    # exit()
    # Update observed reward and arm probabilities
    self.acc_history.append(eval_acc)
    if len(self.pull_history) > 0:
      self.update_vars(self.pull_history[-1])
    # Sample an arm



    arm = np.random.choice(range(self.n_arms), p=self.p)
    self.pull_history.append(arm)
    # kwargs['N'] = N
    # kwargs['already_selected'] = already_selected

    # print("use arm {}".format(arm))
    sample = self.samplers[arm].select_batch(wraped_feat)
    return sample

  def to_dict(self):
    output = {}
    output['samplers'] = self.base_samplers
    output['arm_probs'] = self.probs
    output['pull_history'] = self.pull_history
    output['rewards'] = self.acc_history
    return output

