import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    This module is the inferface for reinforcement learning modelm, which is used to optimize the learning process

    MetaLearningModel(nn.Module):
        A meta-learning model that uses an actor-critic architecture for optimizing a learning process.
        The actor predicts actions based on loss, and the critic predicts rewards for those actions.

        Methods:
            __init__(tokenizer):
                Initializes the MetaLearningModel model with the given tokenizer.
            predict_action(loss):
                Predicts actions based on the input loss using the actor network. Returns the sampled actions
                and their entropy.
            predict_reward(loss, samples):
                Predicts rewards for the given loss and sampled actions using the critic network.
            learning(state, real_reward):
                Performs a learning step for both the actor and critic networks using the given state
                and real reward.
"""

class MetaLearningModel(nn.Module):
    def __init__(self,tokenizer):
        super(MetaLearningModel, self).__init__()

        self.actor = nn.Linear(len(tokenizer.idx2token)+1,(len(tokenizer.idx2token)+1))
        self.critic = nn.Conv1d(2,1,10)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=0.002, betas=(0.9, 0.99), eps=0.0000001)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=0.002, betas=(0.9, 0.99), eps=0.0000001)

    def predict_action(self,loss):

        score = self.actor(loss)
        prob = torch.distributions.bernoulli.Bernoulli(torch.sigmoid(score))
        samples = prob.sample()
        entropy = prob.entropy()

        return samples, entropy

    def predict_reward(self,loss,samples):
        input = torch.cat((loss.unsqueeze(1),samples.view(-1,1,loss.size()[-1])),dim=1)
        reward = self.critic(input).squeeze(1)
        reward = torch.mean(reward)
        return reward

    def learning(self,state,real_reward):

        self.critic_opt.zero_grad()
        action,_ = self.predict_action(state)
        pre_reward = self.predict_reward(state,action)
        loss = F.mse_loss(pre_reward,real_reward)
        loss.backward()
        self.critic_opt.step()

        self.actor_opt.zero_grad()
        action,entropy = self.predict_action(state)
        pre_reward = self.predict_reward(state,action)
        loss = -pre_reward - 0.1*entropy
        loss.backward()
        self.actor_opt.step()