#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 13:31:57 2022

@author: mattgevercer
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('gym_cliffwalking:cliffwalking-v0')

# print(env.action_space) #see action space
# print(env.observation_space) #see observation space

class EpsilonGreedyAgent():
    def __init__(self, epsilon, env): #initialize with environment variable
        # self.FirstStep = True    
        self.epsilon = epsilon
        self.envionment = env
        self.q_dict = {0:-(2**32)}
        self.state_num = {0:1}#by time of first update, will have visted 0 once
        
        self.neighb = {} #{node: [neighbors]}
        self.path = [0] #path of cliffwalker, empties at death and success
        self.map = {} #{(node, target): transition}
        self.move_opps ={0:2, 1:3, 2:0, 3:1} #left matched with right, up matched with down
        
        self.prev_action = None
    
    def action(self):   
        if observation not in self.neighb.keys(): #random action if on a new node
            act = self.envionment.action_space.sample()
            self.prev_action = act
            return act
        else: #choose action based on E-Greedy algo
            if np.random.uniform() > self.epsilon:
                new = dict((k, self.q_dict[k]) for k in self.neighb[observation]) #choose max only  from node neighbors
                max_n = max(new,key=new.get) #neighbor with max q
                self.prev_action = self.map[(observation, max_n)] #keep track of the action
                return self.prev_action 
            else:
                self.prev_action = self.envionment.action_space.sample()
                return self.prev_action
   
    def update(self, reward, obs):#need observation-reward pairs for update
        prev_obs = self.path[-1]
        self.path.append(obs) #update path
        #update state_num to count # of times per state
        if obs not in self.state_num.keys():
            self.state_num.update({obs: 1})
        else:
            self.state_num[obs]+=1
        #update q_dict if on new node
        if obs not in self.q_dict.keys(): 
            self.q_dict.update({obs:reward})
        # mult = np.arange(0,1,1/len(self.path))
        if not done:
            reval = dict(
                (k, 
                 (self.q_dict[k]+(1/(self.state_num[k]))*(reward-self.q_dict[k]))) for k in self.path)  
            self.q_dict.update(reval)#update q_dict 
        if reward != -100 and not done:
            #update neighbor values and keys
            if obs not in self.neighb.keys():
                self.neighb.update({obs:[prev_obs]}) # add obs to neighb with prev obs if not there already
            if obs not in self.neighb[prev_obs]:#check if obs is listed as neighbor to previous node
                a = self.neighb[prev_obs]
                a.append(obs)
                inter = {prev_obs : a}
                self.neighb.update(inter)
            #update map 
            if (prev_obs, obs) not in self.map.values():
                self.map.update({(prev_obs, obs): self.prev_action})
                self.map.update({(obs, prev_obs): self.move_opps[self.prev_action]}) #add the reverse move
        elif done:
            # agent.epsilon *= 0.9995
            mult = np.arange(0,1+(1/len(self.path)),1/len(self.path))
            reval = dict(
                (k, 
                 (self.q_dict[k]+(1/(self.state_num[k]))*(10000*j-self.q_dict[k]))) for k,j in zip(self.path,mult)) 
            self.q_dict.update(reval)#update q_dict 
            self.path = [self.path[0]]
        else:
            self.path = [self.path[0]]


env.action_space.seed(0)
np.random.seed(0)

#Epsilon-Decreasing Strategy
agent = EpsilonGreedyAgent(0.75, env)
won_at = []
epsilon = []

observation = env.reset()
for i_episode in range(10**4):
    observation = env.reset()
    for t in range(100):
        action = agent.action()
        observation, reward, done, info = env.step(action)
        agent.update(reward, observation)
        if done:
            # print("Episode finished after {} timesteps".format(t+1))
            agent.epsilon *= 0.9995
            won_at.append(1)
            break
    if not done:
        won_at.append(0)
    epsilon.append(agent.epsilon)

env.close()

print("# of successful epsiodes for epsilon-decreasing strategy: %d" % (sum(won_at)))
print("The final epsilon value: %3f" % (agent.epsilon))

#Epsilon-Greedy Strategy
agent2 = EpsilonGreedyAgent(0.75, env)
won2 = []

observation = env.reset()
for i_episode in range(10**4):
    observation = env.reset()
    for t in range(100):
        action = agent2.action()
        observation, reward, done, info = env.step(action)
        agent2.update(reward, observation)
        if done:
            # print("Episode finished after {} timesteps".format(t+1))
            won2.append(1)
            break
    if not done:
        won2.append(0)

env.close()
print("# of successful epsiodes for epsilon-greedy strategy: %d" % (sum(won2)))

#Visualize Results
won1 = np.array(won_at)
won2 = np.array(won2)

prop_win1 =  np.sum(won1.reshape(-1, 1000), axis=1)
prop_win2 =  np.sum(won2.reshape(-1, 1000), axis=1)

N=10
ind = np.arange(N)
fig = plt.figure()
ax = fig.add_subplot(111)
width = 0.30 
rects1 = ax.bar(ind, prop_win1, width)
rects2 = ax.bar(ind + width, prop_win2, width)
ax.set_xticks(ind+0.1)
ax.set_xticklabels(('0-1','1-2','2-3','3-4','4-5','5-6','6-7','7-8','8-9','9-10'))
ax.set_xlabel('Episodes (in 1000s)')
ax.set_ylabel('# of Wins')
ax.legend( (rects1[0], rects2[0]), ('\u03B5-Decreasing', '\u03B5-Greedy') )
plt.style.use('fivethirtyeight')
plt.show()



