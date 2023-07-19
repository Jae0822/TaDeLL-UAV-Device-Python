"""
画图：
Fig1:两个CASE的连续三个任务的示意图
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt


with open('Multi Tasks Comp.pkl', 'rb') as f:
        [nTimeUnits, Rewards_Random, Rewards_Random_Natural, Devices_Random, UAV_Random] = pickle.load(f)
name = ['Case 1', 'Case 2']
plt.ion()

num_Devices = 2

fig, ax = plt.subplots(num_Devices, 1, sharex=True)
# fig.suptitle('Learning Process')
for i in range(num_Devices):
    ax[i].plot(np.arange(nTimeUnits), Rewards_Random[i], label="ZSLL")
    ax[i].plot(np.arange(nTimeUnits), Rewards_Random_Natural[i], label="PG")
    ax[i].set_ylabel('Averaged Reward')
    ax[i].set_title(name[i])
    ax[i].legend()
plt.xlabel("Iteration")
# plt.ylabel("common Y")
plt.suptitle("Continuous Environments for Two Cases")
plt.savefig('multi_tasks_temp.eps', format='eps')

d = 1