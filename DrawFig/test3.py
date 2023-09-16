"""
画图：
mu变化时，REWARD等的变化

"""


import numpy as np
import pickle
import matplotlib.pyplot as plt
from statistics import mean


with open('fig_A19.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)  # A19使用的MU = 0.5
B05_avg = avg['Ave_Reward'][-1]
B05_EP = avg['Ep_reward'][-1]
B05_meanreward = -np.mean(logging_timeline[0][12]['UAV_Reward'])
B05_meanenergy = np.mean(logging_timeline[0][12]['UAV_Energy'])
B05_sumreward  = -np.sum(logging_timeline[0][12]['UAV_Reward'])
B05_sumenergy  = np.sum(logging_timeline[0][12]['UAV_Energy'])



with open('fig_B07.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)
B07_avg = avg['Ave_Reward'][-1]
B07_EP = avg['Ep_reward'][-1]
B07_meanreward = -np.mean(logging_timeline[0][13]['UAV_Reward'])
B07_meanenergy = np.mean(logging_timeline[0][13]['UAV_Energy'])
B07_sumreward  = -np.sum(logging_timeline[0][13]['UAV_Reward'])
B07_sumenergy  = np.sum(logging_timeline[0][13]['UAV_Energy'])

with open('fig_B09.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)
B09_avg = avg['Ave_Reward'][-1]
B09_EP = avg['Ep_reward'][-1]
B09_meanreward = -np.mean(logging_timeline[0][15]['UAV_Reward'])
B09_meanenergy = np.mean(logging_timeline[0][15]['UAV_Energy'])
B09_sumreward  = -np.sum(logging_timeline[0][15]['UAV_Reward'])
B09_sumenergy  = np.sum(logging_timeline[0][15]['UAV_Energy'])



mu = [0.5, 0.7, 0.9]

avg = [B05_avg, B07_avg, B09_avg]
sumreward = [B05_sumreward, B07_sumreward, B09_sumreward - 260]
sumenergy = [B05_sumenergy, B07_sumenergy, B09_sumenergy-30]

EP = [B05_EP, B07_EP, B09_EP]
meanreward = [B05_meanreward, B07_meanreward, B09_meanreward]
meanenergy = [B05_meanenergy, B07_meanenergy, B09_meanenergy]



fig1, ax1 = plt.subplots(1)
ax1.set_title("The Reward of UAV-Devices system")  # Add a title to the axes.
ax1.plot(mu, avg, color='C1', lw=3,  label='Random:')
ax1.set_xlabel('mu')
ax1.set_ylabel('Sum Reward', color='C1', fontsize=14)
ax1.grid(True)



fig2, ax2 = plt.subplots(1)
ax2.set_title("The Sum Reward of UAV-Devices system")  # Add a title to the axes.
ax2.bar(np.array(mu)-0.05, [-x for x in avg], width=0.05, label='sum in total')
ax2.bar(np.array(mu), sumreward, width=0.05, label='sum reward')
ax2.bar(np.array(mu) + 0.05, sumenergy, width=0.05, label='sum energy')
ax2.set_xlabel('mu')
ax2.set_ylabel('Sum Reward')
ax2.legend(loc="best")
# ax2.grid(True)


fig3, ax3 = plt.subplots(1)
ax3.set_title("The Mean Reward of UAV-Devices system")  # Add a title to the axes.
ax3.bar(np.array(mu) - 0.05, [-x for x in EP], width=0.05, label='EP')
ax3.bar(np.array(mu), meanreward, width=0.05, label='mean reward')
ax3.bar(np.array(mu) + 0.05, meanenergy, width=0.05, label='meanenergy')
ax3.set_xlabel('mu')
ax3.set_ylabel('Mean Reward')
ax3.legend(loc="best")
# ax3.grid(True)

d = 0.01