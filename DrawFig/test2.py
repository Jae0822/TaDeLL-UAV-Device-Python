"""
画图：UAV速度从10-30，RANDOM和FORCE的REWARD是多少
双纵坐标，两条曲线
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d
import math
import random

from env import Task




random_reward = []
force_reward = []
random_Dcost = []
force_Dcost = []


# velovity = 10
with open('fig_A21.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)
data111 = [-np.mean(env_random.UAV.Reward), -np.mean(env_force.UAV.Reward)]
data22 = [np.mean(env_random.UAV.Energy), np.mean(env_force.UAV.Energy)]
dataAoImean = [np.mean(env_random.UAV.AoI), np.mean(env_force.UAV.AoI)]
dataCPUmean = [np.mean(env_random.UAV.CPU), np.mean(env_force.UAV.CPU)]
random_reward.append((data111[0] + data22[0]) * 0.5)
force_reward.append((data111[1] + data22[1]) * 0.5)
random_Dcost.append((dataAoImean[0] + dataCPUmean[0]) * 0.5)
force_Dcost.append((dataAoImean[1] + dataCPUmean[1]) * 0.5)


# velovity = 15
with open('fig_A19.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)
data111 = [-np.mean(env_random.UAV.Reward), -np.mean(env_force.UAV.Reward)]
data22 = [np.mean(env_random.UAV.Energy), np.mean(env_force.UAV.Energy)]
dataAoImean = [np.mean(env_random.UAV.AoI), np.mean(env_force.UAV.AoI)]
dataCPUmean = [np.mean(env_random.UAV.CPU), np.mean(env_force.UAV.CPU)]
random_reward.append((data111[0] + data22[0]) * 0.5)
force_reward.append((data111[1] + data22[1]) * 0.5)
random_Dcost.append((dataAoImean[0] + dataCPUmean[0]) * 0.5)
force_Dcost.append((dataAoImean[1] + dataCPUmean[1]) * 0.5)


# velovity = 20
with open('fig_A18.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)
data111 = [-np.mean(env_random.UAV.Reward), -np.mean(env_force.UAV.Reward)]
data22 = [np.mean(env_random.UAV.Energy), np.mean(env_force.UAV.Energy)]
dataAoImean = [np.mean(env_random.UAV.AoI), np.mean(env_force.UAV.AoI)]
dataCPUmean = [np.mean(env_random.UAV.CPU), np.mean(env_force.UAV.CPU)]
random_reward.append((data111[0] + data22[0]) * 0.5)
force_reward.append((data111[1] + data22[1]) * 0.5)
random_Dcost.append((dataAoImean[0] + dataCPUmean[0]) * 0.5)
force_Dcost.append((dataAoImean[1] + dataCPUmean[1]) * 0.5)


# velovity = 24
with open('fig_A17.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)
data111 = [-np.mean(env_random.UAV.Reward), -np.mean(env_force.UAV.Reward)]
data22 = [np.mean(env_random.UAV.Energy), np.mean(env_force.UAV.Energy)]
dataAoImean = [np.mean(env_random.UAV.AoI), np.mean(env_force.UAV.AoI)]
dataCPUmean = [np.mean(env_random.UAV.CPU), np.mean(env_force.UAV.CPU)]
random_reward.append((data111[0] + data22[0]) * 0.5)
force_reward.append((data111[1] + data22[1]) * 0.5)
random_Dcost.append((dataAoImean[0] + dataCPUmean[0]) * 0.5)
force_Dcost.append((dataAoImean[1] + dataCPUmean[1]) * 0.5)


# velovity = 36
with open('fig_A13.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)
data111 = [-np.mean(env_random.UAV.Reward), -np.mean(env_force.UAV.Reward)]
data22 = [np.mean(env_random.UAV.Energy), np.mean(env_force.UAV.Energy)]
random_reward.append((data111[0] + data22[0]) * 0.5)
force_reward.append((data111[1] + data22[1]) * 0.5)



fig, ax1 = plt.subplots(1)
ax2 = ax1.twinx()
ax1.set_title("The reward of UAV-Devices system")  # Add a title to the axes.
ax1.plot(np.arange(10, 35, 5), random_reward, color='C1', lw=3,  label='Random:')
ax2.plot(np.arange(10, 35, 5), force_reward,  color='C2', lw=3, label='Force:')
ax1.set_xlabel('UAV Velocity')
ax1.set_ylabel('Random Reward', color='C1', fontsize=14)
ax2.set_ylabel('Force Reward', color='C2', fontsize=14)
ax1.grid(True)

d = 1
