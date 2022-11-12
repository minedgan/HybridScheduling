# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
##### Soft Actor-Critic Algorithm implementation can be found at: https://github.com/pranz24/pytorch-soft-actor-critic
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
          self.action_scale = torch.tensor(1.)
          self.action_bias = torch.tensor(0.)
        #else:
        #    self.action_scale = torch.FloatTensor(
        #        (action_space.high - action_space.low) / 2.)
        #    self.action_bias = torch.FloatTensor(
        #        (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.03)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
    
import math

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam


class SAC(object):
    def __init__(self, gamma, tau, alpha,target_update_interval, hidden_size, lear, num_inputs, action_space, policy = "Deterministic", automatic_entropy_tuning = False):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.policy_type = policy
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        #print(torch.cuda.is_available())
        #print(torch.version.cuda)
        if torch.cuda.is_available():  
          dev = "cuda:0" 
          print("GPU 0 will be used")
        else:   
          dev = "cpu" 
          print("CPU will be used")
        #self.device = torch.device("cpu")
        self.device = torch.device(dev)

        self.critic = QNetwork(num_inputs, path_num, hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lear)

        self.critic_target = QNetwork(num_inputs, path_num, hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(path_num).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=lear)

            self.policy = GaussianPolicy(num_inputs, path_num, hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lear)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, path_num, hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr= lear)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)  
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        #print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        #print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
#Create link capacities and find path capacities
import numpy as np
from itertools import combinations
from itertools import permutations
from scipy.special import factorial
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import argparse
import datetime
from copy import copy
import pandas as pd
import matplotlib.pyplot as plt

N = 25
episode_lim = 200


def create_channel(N):
  import cmath
  x_coor = 50*np.random.rand(N+2)
  y_coor = 50*np.random.rand(N+2)
  dist = np.zeros((N+2,N+2))
  x1 = -0.10879577215395496
  x2 = 1.0292620779496107
  G = 1e13
  n_pl = 2.1
  sigma_sf = 4.9
  c = 3*1e8
  freq = 73*1e9
  lambda_o = c/freq  #Wavelength
  comp_coeff = cmath.sqrt(-1)
  H = np.zeros((N+2,N+2))
  h = np.zeros((N+2,N+2))+comp_coeff*np.zeros((N+2,N+2))
  for i in range(0,N+2):
      a1 = np.array([x_coor[i],y_coor[i]])
      for j in range(i+1,N+2):
          a2 = np.array([x_coor[j],y_coor[j]])
          dist[i,j] = np.linalg.norm(a1-a2)   
          dist[j,i] = np.linalg.norm(a1-a2) 
          if dist[i,j] < 1:
            h[i,j] = 1e3+comp_coeff*1e3
          else:
            path_loss_db = 20*np.log10((4*np.pi)/lambda_o)+10*n_pl*np.log10(dist[i,j]) #+ sigma_sf
            path_loss = np.power(10, (-path_loss_db/10))*G
            v = np.sqrt(path_loss)/2
            h[i,j] = v+v*(x1+comp_coeff*x2)
          H[i,j] = np.log2(1+(np.linalg.norm(h[i,j])**2))
          H[j,i] = np.log2(1+(np.linalg.norm(h[i,j])**2))
  H = np.tril(H,-1)
  H = H + H.T
  H[0,:] = 0
  H[:,N+1] = 0
  all_channels_new = [H]
  print('Dist: ',np.max(dist))
  tmp = dist[dist >0]
  print('Min Dist: ',np.min(tmp))
  print('Maximum link capacity: ',np.max(H))
  return H,dist

def block_edge(H, dist, bern_block, change,block_prob):
    if change == True:
        bern_block = np.zeros((N+2,N+2))
    for i in range(0,N+2):
        for j in range(0,N+2):  
            if i != j:        
                nonblock_p = 1-block_prob[i,j]#np.exp(-dist[i,j]/500)
                #print(nonblock_p)
                if np.isnan(nonblock_p) == True:
                    print('Nan blockage prob')
                elif nonblock_p > 1:
                    print('Nonblock p is greater than 1: ', nonblock_p)
                    print('Distance: ',dist[i,j])
                if change == True:
                    bern_block[i,j] = np.random.binomial(1,1-nonblock_p) 
                    #bern_block[j,i] = copy(bern_block[i,j])
                else:
                    pass
                if bern_block[i,j] == 1:
                    H[i,j] = 1e-7
                    #H[j,i] = 1e-7
    return H,bern_block

 
def vary_channel(H,dict_edges,sigma = 1):
  all_channels = [H]
  for j in range(0,1):
    for i in range(0,episode_lim-1):
      #all_channels.append(H)
      tmp_number = np.zeros((N+2,N+2))
      for i in dict_edges:
        a = sigma*np.random.randn()+H[i[0],i[1]]
        tmp_number[i[0],i[1]] = a
        while(a < 0):
          a = sigma*np.random.randn()+H[i[0],i[1]]
          tmp_number[i[0],i[1]] = a
        
      tmp_number[0,:] = 0
      tmp_number[:,N+1] = 0
      for i in range(0,N+2):
        tmp_number[i,i] = 0

      all_channels.append(tmp_number)
  return all_channels
  

count = 0
instance_num = 5
avg_tr_step = np.zeros((instance_num))
avg_val_step = np.zeros((instance_num))
eps_rate = np.zeros((instance_num,episode_lim))
val_rate = np.zeros((instance_num,episode_lim))
count_channel = 0
episode_lim_pre = 0

#Find the packet rates of paths
def find_pathcap(all_paths,number,H):
  path_cap = np.zeros((N+2,N+2,number))
  for ind in range(0,number):
    path_cap[:,:,ind] = np.multiply(all_paths[:,:,ind],H)
  path_cap[path_cap == 0] = 1e3
  path_cap = np.ndarray.min(np.ndarray.min(path_cap,axis = 0),axis = 0)
  path_cap[path_cap == 1e3] = 1e-7
  return path_cap

#Find the approximate capacity of the network to calculate the desired rate
def calc_desiredRate(H):
  act_time = cp.Variable((N+2,N+2))
  flow = cp.Variable((N+2,N+2))


  obj = 0
  for i in range(0,N+1):
      obj += flow[N+1,i]

                
  objective = cp.Maximize(obj)
  constraints = [flow >=0, act_time >= 0]

  for i in range(0,N+2):
      for j in range(0,N+2):
          constraints.append(flow[j,i] <= act_time[j,i]*H[j,i])
          
        # constraints.append(flow_avg[j,i] <= flow[j,i])
          
  for i in range(0,N+1):
      constraints.append(sum(act_time[:,i]) <= 1)
  for i in range(1,N+2):
      constraints.append(sum(act_time[i,:]) <= 1)
     
  for i in range(1,N+1):
      constraints.append(sum(flow[:,i]) == sum(flow[i,:]))
      #constraints.append(sum(flow_avg[:,i]) == sum(flow_avg[i,:]*(1-block_prob[i,:])))
      
      
  prob = cp.Problem(objective, constraints)
  result_flow = prob.solve(cp.MOSEK,bfs = True) ###Simplex version
  return result_flow
  
### Select initial paths 
def select_initial(H,dist,block_coeff_tmp,dijkstra = False,simplex = True):
  block_prob = np.zeros((N+2,N+2))
  for i in range(0,N+2):
      for j in range(0,N+2):
          block_prob[i,j] = 1-np.exp(-dist[i,j]/block_coeff_tmp[i,j]) #1-np.exp(-dist[i,j]/300) #1/np.random.randint(100,200,1) 
  
  if dijkstra:
    all_paths, num_paths = path_selection(copy(H),block_prob)
    all_paths = all_paths[:,:,0:num_paths]
   

  path_cap = find_pathcap(all_paths,num_paths,H)
  success_prob = np.ones((num_paths))
  for k in range(0,num_paths):
      for i in range(0,N+2):
          for j in range(0,N+2):
              if all_paths[i,j,k] == 1:
                  success_prob[k] *= 1-block_prob[i,j]

  tmp = success_prob[success_prob >0]

  act_time_path = cp.Variable((num_paths))
  obj = 0
  for i in range(0,num_paths):
      obj += act_time_path[i]*path_cap[i]*success_prob[i]

  objective = cp.Maximize(obj)
  constraints = [act_time_path >= 0]
  for i in range(0,N+2):
      coeff1 = np.zeros((num_paths))
      coeff2 = np.zeros((num_paths))
      for j in range(0,num_paths):
          if np.sum(all_paths[:,i,j]) > 0 or np.sum(all_paths[i,:,j]) > 0:
              if 0<=i and i < N+1:
                  ind = np.argwhere(all_paths[:,i,j] == 1)[0][0]
                  coeff1[j] = path_cap[j]/H[ind,i]
              
              if 1<=i and i <= N+1:
                  ind = np.argwhere(all_paths[i,:,j] == 1)[0][0]
                  coeff2[j] = path_cap[j]/H[i,ind]
            
      constraints.append(act_time_path@coeff1 <= 1)
      constraints.append(act_time_path@coeff2 <= 1)

  prob = cp.Problem(objective, constraints)
  if simplex == True:
    res = prob.solve(cp.MOSEK,bfs = True)
  else:
    res = prob.solve()

  act = act_time_path.value
  path_num = np.sum(act>0) 
  rates = np.multiply(path_cap,success_prob)
  rates = np.multiply(rates,act)
  selected_paths = np.argsort(rates)[::-1][:path_num]
  init_state = act[selected_paths]
  return path_num,selected_paths,all_paths[:,:,selected_paths],init_state,result_flow,res#,block_prob,success_prob.std(),success_prob


threshold = 1e-8
result_flow = 1e2
res_path = 0
path_num = 0
std = 0

#Check if a path is already added
def checkSamePath(all_paths1,all_paths2,cnt_path):
    for r in range(0,cnt_path):
        if np.array_equal(all_paths1,all_paths2[:,:,r]):
            return True
    return False

def check_otherpaths(j,prev,selected_nodes,all_paths,dict_edges,cnt_path):
  len_paths = copy(cnt_path)
  k = 0
  while(k<cnt_path):
    print('k: ',k)
    if np.sum(all_paths[prev,:,k]) > 0 or prev == 0:
      l = 0
      while(l < cnt_path):
        if np.sum(all_paths[:,j,l]) > 0 or j == N+1:
          tmp = 0
          next_node = np.argwhere(all_paths[:,0,k] == 1)[0][0]
          list1 = []
          checkValid = True
          while(tmp != prev):
            list1.append(next_node)
            all_paths[next_node,tmp,cnt_path] = 1
            tmp = copy(next_node)
            if next_node != prev:
              next_node = np.argwhere(all_paths[:,next_node,k] == 1)[0][0]
          
          all_paths[j,prev,cnt_path] = 1
          if j != N+1:
            tmp = np.argwhere(all_paths[:,j,l] == 1)[0][0]
          prev_node = copy(j)
          while(prev_node != N+1):
            if prev_node in list1:
              checkValid = False
            all_paths[tmp,prev_node,cnt_path] = 1
   
            prev_node = copy(tmp)
            if tmp != N+1:
              tmp = np.argwhere(all_paths[:,tmp,l] == 1)[0][0]
          
          if checkSamePath(all_paths[:,:,cnt_path],all_paths,cnt_path) == False and checkValid == True:
            cnt_path += 1
          else:
            all_paths[:,:,cnt_path] = np.zeros((N+2,N+2))

        l += 1
    k += 1
    print('Number of paths: ',cnt_path)
  return all_paths,dict_edges,cnt_path


def create_paths2(selected_nodes,all_paths,dict_edges,cnt_path):
    print('Create path!')
    print('Number of paths: ',cnt_path)
    prev = 0
    for j in selected_nodes:
        all_paths[j,prev,cnt_path] = 1
        dict_edges[(j,prev)] += 1
        prev = copy(j)
    all_paths[N+1,j,cnt_path] = 1
    dict_edges[(N+1,j)] += 1
    if checkSamePath(all_paths[:,:,cnt_path],all_paths,cnt_path) == False:
        cnt_path += 1
        prev2 = 0
        for j in selected_nodes:
            if dict_edges[(j,prev2)] == 1:
                all_paths,dict_edges,cnt_path = check_otherpaths(copy(j),copy(prev2),selected_nodes,all_paths,dict_edges,cnt_path)
            prev2 = copy(j)

        if dict_edges[(N+1,j)] == 1:
            all_paths,dict_edges,cnt_path = check_otherpaths(copy(N+1),copy(j),selected_nodes,all_paths,dict_edges,cnt_path)
      
    else:
        all_paths[:,:,cnt_path] = np.zeros((N+2,N+2))
        prev = 0
        for j in selected_nodes:
          dict_edges[(j,prev)] -= 1
          prev = copy(j)
        dict_edges[(N+1,j)] -= 1
    
    return all_paths,dict_edges,cnt_path

def create_channel2(dict_edges):
    import cmath
    x_coor = 100*np.random.rand(N+2)
    y_coor = 100*np.random.rand(N+2)
    dist = np.zeros((N+2,N+2))
    x1 = -0.10879577215395496
    x2 = 1.0292620779496107
    G = 1e12
    n_pl = 2.1
    sigma_sf = 4.9
    c = 3*1e8
    freq = 73*1e9
    lambda_o = c/freq  #Wavelength
    comp_coeff = cmath.sqrt(-1)
    H = np.zeros((N+2,N+2))
    h = np.zeros((N+2,N+2))+comp_coeff*np.zeros((N+2,N+2))
    for i in dict_edges:
        a1 = np.array([x_coor[i[0]],y_coor[i[0]]])
        a2 = np.array([x_coor[i[1]],y_coor[i[1]]])
        dist[i[0],i[1]] = np.linalg.norm(a1-a2)   
        
        if dist[i[0],i[1]] < 1:
            h[i[0],i[1]] = 1e3+comp_coeff*1e3
        else:
            path_loss_db = 20*np.log10((4*np.pi)/lambda_o)+10*n_pl*np.log10(dist[i[0],i[1]]) #+ sigma_sf
            path_loss = np.power(10, (-path_loss_db/10))*G
            v = np.sqrt(path_loss)/2
            h[i[0],i[1]] = v+v*(x1+comp_coeff*x2)
            H[i[0],i[1]] = np.log2(1+(np.linalg.norm(h[i[0],i[1]])**2))
           
 
    H[0,:] = 0
    H[:,N+1] = 0
    for i in range(0,N+2):
      H[i,i] = 0
    all_channels_new = [H]

    return H, dist

#Finds the initial schedule of Baseline 1
def find_initial_schedule(indices,path_cap_withoutdijkstra):
  number_paths = len(indices)
  act_time_path = cp.Variable((number_paths))
  obj = 0
  for i in range(0,number_paths):
      obj += act_time_path[i]*path_cap_withoutdijkstra[indices[i]]

  objective = cp.Maximize(obj)
  constraints = [act_time_path >= 0]
  for i in range(0,N+2):
      coeff1 = np.zeros((number_paths))
      coeff2 = np.zeros((number_paths))
      for j in range(0,number_paths):
          if np.sum(all_paths_withoutdijkstra[:,i,indices[j]]) > 0 or np.sum(all_paths_withoutdijkstra[i,:,indices[j]]) > 0:        
                if 0<=i and i < N+1:
                    ind = np.argwhere(all_paths_withoutdijkstra[:,i,indices[j]] == 1)[0][0]
                    coeff1[j] = path_cap_withoutdijkstra[indices[j]]/H[ind,i]
               
                if 1<=i and i <= N+1:
                    ind = np.argwhere(all_paths_withoutdijkstra[i,:,indices[j]] == 1)[0][0]
                    coeff2[j] = path_cap_withoutdijkstra[indices[j]]/H[i,ind]
               
      constraints.append(act_time_path@coeff1 <= 1)
      constraints.append(act_time_path@coeff2 <= 1)
   

  prob = cp.Problem(objective, constraints)
  res_path = prob.solve(cp.MOSEK,bfs = True)
  print('App capacity: ',res_path)
  act = act_time_path.value
  print('Number of active paths: ',np.sum(act>0))
  return res_path, act

#Finds the initial schedule of Baseline 2
def find_initial_schedule2(indices,path_cap_withoutdijkstra,desired_rate):
  number_paths = len(indices)
  act_time_path = cp.Variable((number_paths))
  obj = 0
  for i in range(0,number_paths):
      obj += act_time_path[i]*path_cap_withoutdijkstra[indices[i]]

  objective = cp.Maximize(0)
  constraints = [act_time_path >= 0]
  for i in range(0,N+2):
      coeff1 = np.zeros((number_paths))
      coeff2 = np.zeros((number_paths))
      for j in range(0,number_paths):
          if np.sum(all_paths_withoutdijkstra[:,i,indices[j]]) > 0 or np.sum(all_paths_withoutdijkstra[i,:,indices[j]]) > 0:        
                if 0<=i and i < N+1:
                    ind = np.argwhere(all_paths_withoutdijkstra[:,i,indices[j]] == 1)[0][0]
                    coeff1[j] = path_cap_withoutdijkstra[indices[j]]/H[ind,i]
                
                if 1<=i and i <= N+1:
                    ind = np.argwhere(all_paths_withoutdijkstra[i,:,indices[j]] == 1)[0][0]
                    coeff2[j] = path_cap_withoutdijkstra[indices[j]]/H[i,ind]
                
      constraints.append(act_time_path@coeff1 <= 1)
      constraints.append(act_time_path@coeff2 <= 1)
      constraints.append(obj == desired_rate)
     

  prob = cp.Problem(objective, constraints)
  res_path = prob.solve(cp.MOSEK,bfs = True)
  print('App capacity: ',res_path)
  act = act_time_path.value
  print('Number of active paths: ',np.sum(act>0))
  return res_path, act


V = np.arange(0,N+2,1)
def select_best_path(H,block_prob):
    R = []
    weight = [-1e8]*(N+2)
    capacity = [-1e8]*(N+2)
    success = [1]*(N+2)
    prev = [np.NAN]*(N+2)
    weight[0] = 1e8
    capacity[0] = 1e8
    success[0] = 1
    
    while(len(R) != N+2):
        max_val = -1e30
        
        for j in V:
            if j not in R:
                if weight[j] > max_val:
                    max_val = weight[j]
                    u = j
  
        R.append(u)
        neighbors = np.argwhere(H[:,u]>0)        
        for v in neighbors:
   
            if v[0] not in R: 
                a = success[u]*(1-block_prob[v[0],u])
                b = min(capacity[u],H[v[0],u])
                c = max(weight[v[0]],a*b)
               
                if c > weight[v[0]]:
                    weight[v[0]] = c
                    success[v[0]] = a
                    capacity[v[0]] = b
                    prev[v[0]] = u
    
                
    return weight, prev
        
####Path Selection Algorithm (modified version of Dijkstra's algorithm)
def path_selection(H, block_prob): 
    print('path selection')
    all_paths = np.zeros((N+2,N+2,5*N))
    num_paths = 0
    while(num_paths < 5*N):
        weight, prev = select_best_path(H, block_prob)
        node = N+1
        min_val = 1e10
        while(node != 0): 
            if math.isnan(prev[node]):
                all_paths[:,:,num_paths] = 0
                return all_paths, num_paths
            
            all_paths[node,prev[node],num_paths] = 1
            if H[node, prev[node]]*(1-block_prob[node, prev[node]]) < min_val:
                min_val = H[node, prev[node]]*(1-block_prob[node, prev[node]]) 
                selected_edge_dest = copy(node)
                selected_edge_source =prev[node]
            node = copy(prev[node])
        
        num_paths += 1
        H[selected_edge_dest, selected_edge_source] = 0
        
    return all_paths, num_paths

def create_network():
    num_paths = 5000
    all_paths = np.zeros((N+2,N+2,num_paths))
    dict_edges = collections.defaultdict(int)
    dict_nodes = collections.defaultdict(int)
    cnt_path = 0
    loop_ind = 0
 
    while(cnt_path < 1000):
        loop_ind += 1
        subset_nodes = np.random.randint(1,10,1)[0]
        selected_nodes = np.array(random.sample(range(1, N+1), subset_nodes))
        
        for sel_ind in selected_nodes:
            dict_nodes[sel_ind] += 1
 
        print('selected nodes: ',selected_nodes)
        all_paths,dict_edges,cnt_path = create_paths2(selected_nodes,all_paths,dict_edges,cnt_path)
    selected_nodes = []
    for i in range(1,N+1):
        if dict_nodes[i] == 0:
            selected_nodes.append(i)

    if selected_nodes != []:
        all_paths,dict_edges,cnt_path = create_paths2(selected_nodes,all_paths,dict_edges,cnt_path)
    print('dict: ',dict_edges)
    all_paths = all_paths[:,:,0:cnt_path]
    num_paths = cnt_path
    print('Number of paths: ',num_paths)
    return all_paths,num_paths,dict_edges
 
#############################################
import collections
import random
import pickle


#'''

####Use the below function to create channels and other variables based on paths that you generated
def create_network_and_channel(title_edges='dict_edges_0',title_channel = 'channels',title_dist = 'dist',title_path = 'all_paths_withoutdijkstra_0',title_block='block_coeffs',title_comp_paths='comp_paths',title_path_selection_channels='path_selection_channels_0',title_block_initial='block_coeffs_initial'):
    with open(title_edges,'rb') as f:
      dict_edges = pickle.load(f)

    with open(title_path,'rb') as f:
      all_paths_withoutdijkstra = pickle.load(f) 

    with open(title_block_initial,'rb') as f:
      block_coeff_list = pickle.load(f) 

    num_paths_withoutdijkstra = all_paths_withoutdijkstra.shape[2]
    with open(title_channel,'rb') as f:
      all_channels_new = pickle.load(f) 
    with open(title_dist,'rb') as f:
      dist = pickle.load(f) 
    H = all_channels_new[0]

    all_channels_new = vary_channel(H,dict_edges)

    with open(title_channel+'_varied','wb') as f:
        pickle.dump(all_channels_new,f)
#####



    block_coeff = copy(block_coeff_list[0]) 
    block_prob = np.zeros((N+2,N+2))
    for i in range(0,N+2):
        for j in range(0,N+2):
            block_prob[i,j] = 1-np.exp(-dist[i,j]/block_coeff[i,j])
###3
    all_paths_tmp, num_paths_tmp = path_selection(copy(H),block_prob)
    all_paths_tmp = all_paths_tmp[:,:,0:num_paths_tmp]
    print('Number of paths: ',num_paths_tmp)
####

    all_paths = all_paths_tmp
    num_paths = num_paths_tmp
    success_prob_withoutdijkstra = np.ones((num_paths_withoutdijkstra))
    for k in range(0,num_paths_withoutdijkstra):
        for i in range(0,N+2):
            for j in range(0,N+2):
                if all_paths_withoutdijkstra[i,j,k] == 1:# or all_paths[j,i,k] == 1:
                    success_prob_withoutdijkstra[k] *= 1-block_prob[i,j]
    print('shape of success prob: ',success_prob_withoutdijkstra)

    success_prob = np.ones((num_paths))
    for k in range(0,num_paths):
        for i in range(0,N+2):
            for j in range(0,N+2):
                if all_paths[i,j,k] == 1:# or all_paths[j,i,k] == 1:
                    success_prob[k] *= 1-block_prob[i,j]

    union_paths = 0
    union = []
    init_union = []
    block_coeff_list = []
    comp_paths = np.zeros((N+2,N+2,10000))
    paths_found = True
    trial_num = 0
    path_selection_channels = []#channels that are used for selecting different paths and taking union of them
    while(union_paths < 10 and trial_num < 20):
        trial_num += 1
        if union_paths > 0:
            all_channels_varied_tmp = vary_channel(H,dict_edges)
            H_varied_tmp = all_channels_varied_tmp[1]
            block_coeff_tmp = 50+700*np.random.rand(N+2,N+2)
        else:
            block_coeff_tmp = block_coeff
            H_varied_tmp = copy(H)

        path_num, selected_paths,returned_paths,init_state_tmp,result_flow,res = select_initial(H_varied_tmp,dist,block_coeff_tmp,dijkstra = True)
        if union_paths == 0:
            init_state = init_state_tmp
        for i in range(0,path_num):
            if union_paths == 0 or checkSamePath(returned_paths[:,:,i],comp_paths,union_paths) == False:#selected_paths[i] not in union:
                block_coeff_list.append(block_coeff_tmp)
                path_selection_channels.append(H_varied_tmp)
                union.append(selected_paths[i])
                comp_paths[:,:,union_paths] = returned_paths[:,:,i]
                init_union.append(init_state_tmp[i])
                union_paths +=1 
                print('union paths: ',union_paths)
    if union_paths < 10:
      paths_found = False
      return 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,paths_found
    selected_paths = union
    init_state = init_union
    path_num = union_paths
    with open(title_block,'wb') as f:
        pickle.dump(block_coeff_list,f)

    with open(title_path_selection_channels,'wb') as f:
        pickle.dump(path_selection_channels,f)

    comp_paths = comp_paths[:,:,0:union_paths]
    with open(title_comp_paths,'wb') as f:
        pickle.dump(comp_paths,f)

    print('number of selected paths: ',path_num)
    print('shape of comp paths: ',comp_paths.shape)
  
    all_paths = all_paths_tmp
    num_paths = num_paths_tmp
    sel_cap = find_pathcap(comp_paths,path_num,H)
    sel_cap_init = sel_cap
    print('number of paths: ',num_paths)

    return H,all_channels_new,dist,dict_edges,all_paths_withoutdijkstra,num_paths_withoutdijkstra,block_coeff,block_prob,success_prob_withoutdijkstra,all_paths,num_paths,success_prob,selected_paths,init_state,path_num,sel_cap,sel_cap_init,comp_paths,paths_found
#'''
#Uncomment the below part to generate a new set of topology
'''
for i in range(0,instance_num):
  title_path = 'all_paths_withoutdijkstra_'+str(i)
  title_edges = 'dict_edges_'+str(i)
  all_paths_withoutdijkstra, num_paths_withoutdijkstra,dict_edges = create_network()
  with open(title_path,'wb') as f:
      pickle.dump(all_paths_withoutdijkstra,f)
  with open(title_edges,'wb') as f:
      pickle.dump(dict_edges,f)

found_paths = False
while(found_paths == False):
  H,all_channels_new,dist,dict_edges,all_paths_withoutdijkstra,num_paths_withoutdijkstra,block_coeff,block_prob,success_prob_withoutdijkstra,all_paths,num_paths,success_prob,selected_paths,init_state,path_num,sel_cap,sel_cap_init,comp_paths,found_paths = create_network_and_channel()
'''
#'''
###Choose a new set of initial paths
def new_selected_paths(prev_path_num,block_coeffs_list,path_selection_channels_list,comp_paths_list,dict_edges,all_paths_withoutdijkstra,H,dist,block_coeff,block_prob):
    num_paths_withoutdijkstra = all_paths_withoutdijkstra.shape[2]
    

###3
    all_paths_tmp, num_paths_tmp = path_selection(copy(H),block_prob)
    all_paths_tmp = all_paths_tmp[:,:,0:num_paths_tmp]
#    print('Number of paths: ',num_paths_tmp)
####

    all_paths = all_paths_tmp
    num_paths = num_paths_tmp

    union_paths = 0
    union = []
    init_union = []
    
    comp_paths = np.zeros((N+2,N+2,10000))
    paths_found = True
    trial_num = 0
    block_coeffs_list_tmp = []
    path_selection_channels_list_tmp = []
    while(union_paths < prev_path_num and trial_num < 20):
        trial_num += 1
        if union_paths > 0:
            all_channels_varied_tmp = vary_channel(H,dict_edges)
            H_varied_tmp = all_channels_varied_tmp[1]
            block_coeff_tmp = 50+700*np.random.rand(N+2,N+2)#1.5*block_coeff*np.random.rand()
        else:
            block_coeff_tmp = block_coeff
            H_varied_tmp = copy(H)

        path_num, selected_paths,returned_paths,init_state_tmp,result_flow,res = select_initial(H_varied_tmp,dist,block_coeff_tmp,dijkstra = True)
        if union_paths == 0:
            init_state = init_state_tmp
        for i in range(0,path_num):
            if union_paths == 0 or checkSamePath(returned_paths[:,:,i],comp_paths,union_paths) == False:#selected_paths[i] not in union:
                block_coeffs_list_tmp.append(block_coeff_tmp)
                path_selection_channels_list_tmp.append(H_varied_tmp)
                union.append(selected_paths[i])
                comp_paths[:,:,union_paths] = returned_paths[:,:,i]
                init_union.append(init_state_tmp[i])
                union_paths +=1 
                if union_paths == prev_path_num:
                    break
#                print('union paths: ',union_paths)
    if union_paths < prev_path_num:
      paths_found = False
      return 0,0,0,0,0,paths_found
    selected_paths = union
    init_state = init_union
    path_num = union_paths

    comp_paths = comp_paths[:,:,0:union_paths]
    comp_paths_list.append(comp_paths)
    block_coeffs_list.append(block_coeffs_list_tmp)
    path_selection_channels_list.append(path_selection_channels_list_tmp)


    all_paths = all_paths_tmp
    num_paths = num_paths_tmp
    sel_cap = find_pathcap(comp_paths,path_num,H)
    sel_cap_init = sel_cap
#    print('number of paths: ',num_paths)
  
    return selected_paths,path_num,sel_cap,sel_cap_init,comp_paths,paths_found

#'''

#'''
def new_selected_paths_retrieved(comp_paths_list_saved,selection_num, prev_path_num,block_coeffs_list,path_selection_channels_list,comp_paths_list,dict_edges,all_paths_withoutdijkstra,H,dist,block_coeff,block_prob):
    num_paths_withoutdijkstra = all_paths_withoutdijkstra.shape[2]
###3
    all_paths_tmp, num_paths_tmp = path_selection(copy(H),block_prob)
    all_paths_tmp = all_paths_tmp[:,:,0:num_paths_tmp]
####

    all_paths = all_paths_tmp
    num_paths = num_paths_tmp

    comp_paths = comp_paths_list_saved[selection_num]
    path_num = comp_paths.shape[2]
    all_paths = all_paths_tmp
    num_paths = num_paths_tmp
    sel_cap = find_pathcap(comp_paths,path_num,H)
    sel_cap_init = sel_cap
  
    return path_num,sel_cap,sel_cap_init,comp_paths
#'''


#'''
#Retrieve the topology and the channel used in the paper
def retrieve_channels(title_edges='dict_edges_0',title_channel = 'channels',title_dist = 'dist',title_path = 'all_paths_withoutdijkstra_0',title_block='block_coeffs',title_comp_paths='comp_paths',title_path_selection_channels='path_selection_channels_0',title_block_initial='block_coeffs_initial',order = 0):
    with open(title_edges,'rb') as f:
      dict_edges = pickle.load(f)

    with open(title_path,'rb') as f:
      all_paths_withoutdijkstra = pickle.load(f) 

    with open(title_block_initial,'rb') as f:
      block_coeff_list = pickle.load(f) 

    num_paths_withoutdijkstra = all_paths_withoutdijkstra.shape[2]
    with open(title_channel,'rb') as f:
      all_channels_new = pickle.load(f) 
    with open(title_dist,'rb') as f:
      dist = pickle.load(f) 
    with open(title_channel+'_varied','rb') as f:
      all_channels_new = pickle.load(f) 
    H = all_channels_new[0]
 

    block_coeff = copy(block_coeff_list[0]) 
    block_prob = np.zeros((N+2,N+2))
    for i in range(0,N+2):
        for j in range(0,N+2):
            block_prob[i,j] = 1-np.exp(-dist[i,j]/block_coeff[i,j])
###3
    all_paths_tmp, num_paths_tmp = path_selection(copy(H),block_prob)
    all_paths_tmp = all_paths_tmp[:,:,0:num_paths_tmp]
    print('Number of paths: ',num_paths_tmp)
####

    all_paths = all_paths_tmp
    num_paths = num_paths_tmp
    success_prob_withoutdijkstra = np.ones((num_paths_withoutdijkstra))
    for k in range(0,num_paths_withoutdijkstra):
        for i in range(0,N+2):
            for j in range(0,N+2):
                if all_paths_withoutdijkstra[i,j,k] == 1:# or all_paths[j,i,k] == 1:
                    success_prob_withoutdijkstra[k] *= 1-block_prob[i,j]
    print('shape of success prob: ',success_prob_withoutdijkstra)

    success_prob = np.ones((num_paths))
    for k in range(0,num_paths):
        for i in range(0,N+2):
            for j in range(0,N+2):
                if all_paths[i,j,k] == 1:# or all_paths[j,i,k] == 1:
                    success_prob[k] *= 1-block_prob[i,j]
    print('Hi')
    with open(title_block,'rb') as f:
      block_coeff_list = pickle.load(f) 

    with open(title_comp_paths,'rb') as f:
      comp_paths = pickle.load(f) 
    path_num = comp_paths.shape[2]

    all_paths = all_paths_tmp
    num_paths = num_paths_tmp
    sel_cap = find_pathcap(comp_paths,path_num,H)
    sel_cap_init = sel_cap
    print('number of paths: ',num_paths)
  
    return H,all_channels_new,dist,dict_edges,all_paths_withoutdijkstra,num_paths_withoutdijkstra,block_coeff,block_prob,success_prob_withoutdijkstra,all_paths,num_paths,success_prob,path_num,sel_cap,sel_cap_init,comp_paths

H,all_channels_new,dist,dict_edges,all_paths_withoutdijkstra,num_paths_withoutdijkstra,block_coeff,block_prob,success_prob_withoutdijkstra,all_paths,num_paths,success_prob,path_num,sel_cap,sel_cap_init,comp_paths = retrieve_channels()
#'''


eps_rate = np.zeros((instance_num,episode_lim+1))
val_rate = np.zeros((instance_num,episode_lim+1))
inv_count = 0
init_states = []
finish = False
avg_des = np.zeros((episode_lim+1))
invalid_training = np.zeros((instance_num,episode_lim))
invalid_validation = np.zeros((instance_num,episode_lim))
benchmark_rates = np.zeros((instance_num,episode_lim+1))
benchmark_rates2 = np.zeros((instance_num,episode_lim+1))
numOfTimesNewPathsSelected = np.zeros((instance_num))
H_placeholder = []
num_episodes = 5
count_lim = 500
val_states = []

for j in range(0,num_episodes):
  valstate = np.zeros((path_num))
  val_states.append(valstate)

###Static network channel
with open('H_blocked_0.7','rb') as f:
  H_placeholder_static = pickle.load(f)

with open('H_blocked','rb') as f:
  H_placeholder_new = pickle.load(f)
episode_count = 0
for instance_ind in range(0,instance_num):
    with open('comp_paths_list_'+str(instance_ind),'rb') as f:
      comp_paths_list_saved = pickle.load(f)
    selection_num = 0
    block_coeffs_list = []
    path_selection_channels_list = []
    new_channels_list = []
    comp_paths_list = []
    channel_count_list = []
    print("Instance: ", instance_ind)
    if instance_ind > 0 and instance_ind %1 == 0:
        #'''
        #The same topology and the channel used in the paper
        title_dist = 'dist_'+str(instance_ind)
        title_channel = 'channels_'+str(instance_ind)
        title_block = 'block_coeffs_'+str(instance_ind)
        title_comp_paths = 'comp_paths_'+str(instance_ind)
        title_path = 'all_paths_withoutdijkstra_'+str(instance_ind)
        title_edges = 'dict_edges_'+str(instance_ind)
        title_path_selection_channels = 'path_selection_channels_'+str(instance_ind)
        title_block_initial='block_coeffs_initial_'+str(instance_ind)
        H,all_channels_new,dist,dict_edges,all_paths_withoutdijkstra,num_paths_withoutdijkstra,block_coeff,block_prob,success_prob_withoutdijkstra,all_paths,num_paths,success_prob,path_num,sel_cap,sel_cap_init,comp_paths = retrieve_channels(title_edges,title_channel,title_dist,title_path, title_block,title_comp_paths,title_path_selection_channels,title_block_initial,instance_ind)
        #'''
        '''
        #Generates a new topology and channel at each instance
        found_paths = False
        while(found_paths == False):
            print("Instance: ", instance_ind)
            title_dist = 'dist_'+str(instance_ind)
            title_channel = 'channels_'+str(instance_ind)
            title_block = 'block_coeffs_'+str(instance_ind)
            title_comp_paths = 'comp_paths_'+str(instance_ind)
            title_path = 'all_paths_withoutdijkstra_'+str(instance_ind)
            title_edges = 'dict_edges_'+str(instance_ind)
            title_path_selection_channels = 'path_selection_channels_'+str(instance_ind)
            title_block_initial='block_coeffs_initial_'+str(instance_ind)
            H,all_channels_new,dist,dict_edges,all_paths_withoutdijkstra,num_paths_withoutdijkstra,block_coeff,block_prob,success_prob_withoutdijkstra,all_paths,num_paths,success_prob,selected_paths,init_state,path_num,sel_cap,sel_cap_init,comp_paths,found_paths = create_network_and_channel(title_edges,title_channel,title_dist,title_path, title_block,title_comp_paths,title_path_selection_channels,title_block_initial)
        '''
    initial_path_num = copy(path_num)
    count_channel = 0
        
    import matplotlib
    matplotlib.use('Agg')
    import argparse
    import datetime
    import numpy as np
    import itertools
    from copy import copy
    import pandas as pd
    import matplotlib.pyplot as plt
    gamma = 1
    tau = 0.005
    lr = 0.0003
    alpha = 0.2
    hidden_size = 256
    updates_per_step = 1
    target_update_interval = 1
    replay_size = 1000000
    seed = 123456
    rates = [4]
    exp_epsilon = 0.01

    decay = 1
    batch_size = 32
    step_limits = [500]
    count = 0
        
    avg_rate = np.zeros((len(step_limits)))
    for step_ind in range(0,len(step_limits)):
      print("Step ind: ",step_ind)
      step_lim = int(step_limits[step_ind])

      # Agent
      agent = SAC(gamma, tau, alpha, target_update_interval, hidden_size,lr, path_num, None, policy = "Gaussian",automatic_entropy_tuning = True)

      # Memory
      memory = ReplayMemory(replay_size,seed)

      # Training Loop
      total_numsteps = 0
      updates = 0
      rewards = []
      avg_rewards = []
      temp_rew = []
      temp_rew = []
      found = False
     
      count_channel = 0
      
      print("Instance: ",instance_ind)
      H = all_channels_new[count_channel]
      
      count_channel += 1 
      print(count_channel)

      path_cap = find_pathcap(all_paths,num_paths,H) 
      sel_cap = find_pathcap(comp_paths,path_num,H)

      capacity_app = calc_desiredRate(H)
      des_rate = capacity_app*(0.7)
      path_cap_withoutdijkstra = find_pathcap(all_paths_withoutdijkstra,num_paths_withoutdijkstra,H)
      print('shape of path cap: ',path_cap_withoutdijkstra)
      indices = np.argsort(success_prob_withoutdijkstra)                
      success_path = indices[num_paths_withoutdijkstra-path_num:num_paths_withoutdijkstra]
      capacity_path = np.argsort(path_cap_withoutdijkstra)[num_paths_withoutdijkstra-path_num:num_paths_withoutdijkstra]
      success_and_capacity_path = np.argsort(np.multiply(path_cap_withoutdijkstra,success_prob_withoutdijkstra))[num_paths_withoutdijkstra-path_num:num_paths_withoutdijkstra]
      benchmark_rates[instance_ind,1],success_schedule = find_initial_schedule(success_path,path_cap_withoutdijkstra)
      benchmark_rates2[instance_ind,1],capacity_schedule  = find_initial_schedule(capacity_path,path_cap_withoutdijkstra)
      benchmark_rates3[instance_ind,1], success_and_capacity_schedule = find_initial_schedule(success_and_capacity_path,path_cap_withoutdijkstra)
      benchmark_rates[instance_ind,1],base_schedule = find_initial_schedule(np.arange(0,num_paths_withoutdijkstra),path_cap_withoutdijkstra)
      benchmark_rates2[instance_ind,1], new_schedule = find_initial_schedule2(np.arange(0,num_paths_withoutdijkstra),path_cap_withoutdijkstra,des_rate)
      benchmark_rates2[instance_ind,1] = np.sum(np.multiply(new_schedule,path_cap_withoutdijkstra))
      
      print('Baseline 1 rate: ',benchmark_rates[instance_ind,1])
      print('Baseline 2 rate: ',benchmark_rates2[instance_ind,1])
   
      blockage = True
      blockage2 = False
      des_rate_count = 0
      blockage_time = 0
      
      
      avg_des[0] = avg_des[0]+des_rate
      print('desired rate: ',des_rate)
      blockage_applied = True
      variation = False
      H_old = copy(H)
      bern_block = np.zeros((N+2,N+2))
      blocked_paths = np.zeros((episode_lim))
      blocked_paths2 = np.zeros((episode_lim))
      blocked_links = np.zeros((episode_lim))
      last_rewards_count = 0
      success_prob_comp_paths = np.ones((path_num))
      for k in range(0,path_num):
          for i in range(0,N+2):
              for j in range(0,N+2):
                  if comp_paths[i,j,k] == 1:# or all_paths[j,i,k] == 1:
                      success_prob_comp_paths[k] *= 1-block_prob[i,j]
      exploration_selected_paths = np.argsort(success_prob_comp_paths)[::-1]
      for episode in range(episode_lim):
          if episode == 0:
            H_tmp = copy(H)
          #'''
          if episode > 0:
  
            if instance_ind >= 0:
              H = all_channels_new[count_channel]
              #Block a new set of links at every 10 episodes
              if blockage == True:
                if episode %10 == 0:
                    change = True
                else:
                    change = False
                H_tmp, bern_block = block_edge(copy(H),dist,bern_block,change,block_prob)
                
               
            H_tmp = H_placeholder_new[episode_count]  #To use the same channel matrix as in the paper
            episode_count += 1
       
            #Uncomment the below part to generate a new channel matrix but with the same blockage pattern as the static network case
            '''
            H_tmp2 = H_placeholder_static[episode_count]
            links_blocked = (H_tmp2 == 1e-7)
            H_tmp = copy(H)
            H_tmp[links_blocked] = 1e-7
            H_placeholder.append(H_tmp)
            '''
           
            episode_count += 1
            count_channel += 1 

            path_cap = find_pathcap(all_paths,num_paths,H_tmp)
            path_cap_withoutdijkstra = find_pathcap(all_paths_withoutdijkstra,num_paths_withoutdijkstra, H_tmp)
          
    
            sel_cap = find_pathcap(comp_paths,path_num,H_tmp)
        
            #Baseline methods
            benchmark_rates[instance_ind,episode+1] = np.sum(np.multiply(base_schedule,path_cap_withoutdijkstra))
            benchmark_rates2[instance_ind,episode+1] = np.sum(np.multiply(new_schedule,path_cap_withoutdijkstra))
     
            blocked_paths[episode] = np.sum(sel_cap < 1e-3)
            blocked_paths2[episode] = np.sum(path_cap_withoutdijkstra  < 1e-3)
            blocked_links[episode] = np.sum(H_tmp == 1e-7)
     
             
            capacity_app = calc_desiredRate(H_tmp)
            des_rate = capacity_app*0.7
            if des_rate < 0:
              des_rate = 0

          avg_des[episode+1] = avg_des[episode+1] + des_rate

   
          episode_reward = 0
          episode_steps = 0
          done = False
        
          if episode%10 == 0:
            exp_epsilon = exp_epsilon*decay

 
          
          state = np.zeros((path_num))
          count_invalid_state = 0
          
          for step in range(step_lim):
              #Select an action
              bern_num = np.random.binomial(1,1-exp_epsilon)
              if bern_num == 1:
                action = agent.select_action(state)
              else:
                subset_num = np.random.randint(1,path_num+1,1)[0]
                rand_path = exploration_selected_paths[0:subset_num] 
                action = np.zeros((path_num))
                for subset in rand_path:
                  action[subset] =  np.random.rand()
                
           
              if np.isnan(np.sum(action)):
                print('Invalid action')
                inv_count += 1 
                action = 0.2*np.random.rand(path_num)-0.1
              
              if bern_num == 1:
                action[action < 1e-3] = 0 #Action clipping
    
              temp_new_state = state + action 
              act_times = np.divide(temp_new_state,sel_cap)
              reward = 0
              counted = False
              
              if np.sum(act_times < 0) > 0:
                count_invalid_state = count_invalid_state + 1
                counted = True
              
              finish = False

              #Check if the next state is physically feasible
              node_time_pre = np.zeros((N+2))
              node_time_fol = np.zeros((N+2))
              for node in range(0,N+2):
                for i in range(0,path_num):
                  if np.sum(comp_paths[node,:,i]) > 0 or np.sum(comp_paths[:,node,i]) > 0:
                    if node != 0:
                      ind_pre = np.argwhere(comp_paths[node,:,i] == 1)[0][0]
                      node_time_pre[node] = node_time_pre[node] + act_times[i]*(sel_cap[i]/H_tmp[node,ind_pre])
                    if node != N+1:
                      ind_fol = np.argwhere(comp_paths[:,node,i] == 1)[0][0]
                      node_time_fol[node] = node_time_fol[node] + act_times[i]*(sel_cap[i]/H_tmp[ind_fol,node])
              if np.sum(node_time_pre[1:N+2] > 1) > 0 or np.sum(node_time_fol[0:N+1] > 1) > 0 or np.sum(act_times < 0) > 0 or np.any(temp_new_state > sel_cap):
                if counted == False:
                  count_invalid_state = count_invalid_state + 1              

                new_state = copy(state) ###eski
              else:
                new_state = copy(temp_new_state)

              eps_rate[instance_ind,episode+1] = eps_rate[instance_ind,episode+1] + np.sum(new_state)  ## Average Rate bakarken AC

              #Check if the target packet rate is achieved
              if np.sum(new_state) >= des_rate:
                last_rewards_count = 0
                reward = 1
                finish = True
                eps_rate[instance_ind,episode+1] = (eps_rate[instance_ind,episode+1] + np.sum(new_state)*(step_lim-step-1))/step_lim 
              else:
                reward = np.exp(np.sum(new_state))/10000000
      
                last_rewards_count += 1
              if last_rewards_count >5*step_lim:
                print('NEW PATHS WILL BE SELECTED!')
                numOfTimesNewPathsSelected[instance_ind] += 1
                last_rewards_count = 0
                if selection_num < len(comp_paths_list_saved):
                  path_num,sel_cap,sel_cap_init,comp_paths = new_selected_paths_retrieved(comp_paths_list_saved,selection_num,initial_path_num,block_coeffs_list,path_selection_channels_list,comp_paths_list,dict_edges,all_paths_withoutdijkstra,copy(H_tmp),dist,block_coeff,block_prob)
                  selection_num += 1
                else:
                  found_paths = False
                  new_channels_list.append(H_tmp)
                  channel_count_list.append(count_channel-1)
                  while(found_paths == False):
                      selected_paths,path_num,sel_cap,sel_cap_init,comp_paths,found_paths = new_selected_paths(initial_path_num,block_coeffs_list,path_selection_channels_list,comp_paths_list,dict_edges,all_paths_withoutdijkstra,copy(H_tmp),dist,block_coeff,block_prob)
              if step == step_lim-1 and finish == False:
                temp_rew.append(0)
                eps_rate[instance_ind,episode+1] = eps_rate[instance_ind,episode+1]/step_lim
                
                
              episode_steps += 1
              total_numsteps += 1
              episode_reward += reward

              done = False
              memory.push(state, action, reward, new_state, done) # Append transition to memory
              if len(memory) > batch_size:
                  critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, batch_size, updates)
                  updates += 1
              state = copy(new_state)

              if finish:
                break
      
          ####### Evaluation     
          avgval = np.zeros((path_num))
          lim_val = int(step_limits[step_ind])
          epsilon_valid = 0
          for valid_ind in range(0,num_episodes):
            count_invalid_state_val = 0
            val_rate_ind = 0
            valstate = np.zeros((path_num))
            for valid_step in range(0,lim_val):
              bern_num = np.random.binomial(1,1-epsilon_valid)
              if bern_num == 1:
                action = agent.select_action(valstate)
              else:
                action = np.random.rand(path_num)
              action[action < 1e-3] = 0

              if np.isnan(np.sum(action)):
                print('Invalid action')
                action = 0.2*np.random.rand(path_num)-0.1
    

              temp_new_state = valstate + action
              act_times = np.divide(temp_new_state,sel_cap)
              counted = False
        
              if np.sum(act_times < 0) > 0:
                count_invalid_state_val = count_invalid_state_val + 1
                counted = True
 
              node_time_pre = np.zeros((N+2))
              node_time_fol = np.zeros((N+2))
              for node in range(0,N+2):
                for i in range(0,path_num):
                  if np.sum(comp_paths[node,:,i]) > 0 or np.sum(comp_paths[:,node,i]) > 0:
                    
                    if node != 0:
                      ind_pre = np.argwhere(comp_paths[node,:,i] == 1)[0][0]
                      node_time_pre[node] = node_time_pre[node] + act_times[i]*(sel_cap[i]/H_tmp[node,ind_pre])
                    if node != N+1:
                      ind_fol = np.argwhere(comp_paths[:,node,i] == 1)[0][0]
                      node_time_fol[node] = node_time_fol[node] + act_times[i]*(sel_cap[i]/H_tmp[ind_fol,node])
              if np.sum(node_time_pre[1:N+2] > 1) > 0 or np.sum(node_time_fol[0:N+1] > 1) > 0 or np.sum(act_times < 0) > 0 or np.any(temp_new_state > sel_cap):
                if counted == False:
                  count_invalid_state_val = count_invalid_state_val + 1              
 
                new_state = copy(valstate) ###eski
              else:
                new_state = copy(temp_new_state)
          
              if np.sum(new_state) >= des_rate:
                val_rate[instance_ind,episode+1] = val_rate[instance_ind,episode+1] + np.sum(new_state)
           
                break
              if valid_step == lim_val-1:
         
                val_rate[instance_ind,episode+1] = val_rate[instance_ind,episode+1] + np.sum(new_state)
              valstate = copy(new_state)
               
          val_rate[instance_ind,episode+1] = val_rate[instance_ind,episode+1]/num_episodes
            
          print('instance num: ',instance_ind)
          print('episode: ',episode)
          print('desired rate: ',des_rate)
          print("Evaluation rate: ",val_rate[instance_ind,episode+1])
          print("Training rate: ",eps_rate[instance_ind,episode+1])     
          print("Baseline 1 rate: ",benchmark_rates[instance_ind,episode+1])
          print("Baseline 2 rate: ",benchmark_rates2[instance_ind,episode+1])
          print('Number of invalid states during training: ',(count_invalid_state/(step+1))*100)
          print('Number of invalid states during validation: ',(count_invalid_state_val/(valid_step+1))*100)
          print('Training steps: ',step)
          print('Validation steps: ',valid_step)
          avg_tr_step[instance_ind] += step+1
          avg_val_step[instance_ind] += valid_step+1
          invalid_training[instance_ind,episode] = (count_invalid_state/(step+1))*100
          invalid_validation[instance_ind,episode] = (count_invalid_state_val/(valid_step+1))*100

    
avg_tr_step /= episode_lim
avg_val_step /= episode_lim
print('Average tr: ',np.sum(avg_tr_step)/instance_num)
print('Average val step: ',np.sum(avg_val_step)/instance_num)
eps_rate2 = np.sum(eps_rate,0)/instance_num
val_rate2 = np.sum(val_rate,0)/instance_num
invalid_tr = np.sum(invalid_training,0)/instance_num
invalid_val = np.sum(invalid_validation,0)/instance_num
avg_des = avg_des/instance_num

benchmark = np.sum(benchmark_rates,0)/instance_num
benchmark2 = np.sum(benchmark_rates2,0)/instance_num




episode = np.linspace(0,episode_lim,episode_lim+1)
fig = plt.figure()
plt.plot(episode,eps_rate2, label = 'Average Training Rate')
plt.plot(episode,val_rate2, label = 'Evaluation Rate')
plt.plot(episode,avg_des,label='Desired Rate')
plt.plot(episode,benchmark, label = 'Baseline 1')
plt.plot(episode,benchmark2, label = 'Baseline 2')
plt.xlabel('Episodes')
plt.ylabel('Rate')
plt.legend(loc='upper right')
plt.grid()
fig.savefig('test_blockage.png')
plt.show()



