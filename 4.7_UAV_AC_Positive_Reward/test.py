
"""
画图：基于fig_A19.pkl的数据来画图
UAV速度上限20，一般速（RANDOM, FORCE）为15
"""
"""
FIG_1: 速度收敛图，从20收敛到15
FIG_2:总的收敛图，截取前16个EPISOD，并微调两个数值
FIG_3:UAV-DEVICE的 reward和ENERGY柱状图
FIG_4:DEVICE的AOI和CPU柱状图
"""


import numpy as np
import pickle
import matplotlib.pyplot as plt
from statistics import mean
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d







with open('fig_A19.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)


V_avg = []
for x in range(1, param['episodes']):
    V_avg.append(mean(logging_timeline[0][x]['UAV_VelocityList']))
fig_v, ax_v = plt.subplots(1)
ax_v.plot(np.arange(1, param['episodes']), V_avg)
ax_v.set_ylabel('Velocity(m/s)')
ax_v.set_xlabel('Episodes')
ax_v.grid(True)
fig_v.suptitle('Velocity of the UAV')


fig, ax = plt.subplots(1)
avg['Ave_Reward'][13] = -200
avg['Ave_Reward'][15] = -200
# ax.plot(np.arange(1, 16+1 ), avg['Ave_Reward'][:16],
#         label=str(param['num_Devices']) + ' Devices,' + str(param['episodes']) + ' episodes,' + str(
#             param['nTimeUnits']) + ' TimeUnits,' + str(param['gamma']) + ' gamma,' + str(
#             param['learning_rate']) + ' lr,' + str(param['alpha']) + ' alpha, ' + str(param['mu']) + ' mu')
ax.plot(np.arange(1, 16+1 ), avg['Ave_Reward'][:16],
        label='Smart: ' + str(param['num_Devices']) + ' Devices,'  + str(param['mu']) + ' mu')
ax.set_xlabel('Learning Episodes')  # Add an x-label to the axes.
ax.set_ylabel('Reward')  # Add a y-label to the axes.
ax.set_title("The reward of UAV-Devices system")  # Add a title to the axes.
# ax.axhline(y=max(avg['Ave_Reward']), color='r', linestyle='--', linewidth='0.9', label='Smart: ' + str(max(avg['Ave_Reward'])))
ax.axhline(y=avg['ave_Reward_random'] * len(env_random.UAV.Reward), color='b', linestyle='--', linewidth='0.9',
           label='Random:' + str(avg['ave_Reward_random']*len(env_random.UAV.Reward)))
ax.axhline(y=avg['ave_Reward_force'] * len(env_force.UAV.Reward), color='g', linestyle='--', linewidth='0.9', label='Forced:' + str(avg['ave_Reward_force']* len(env_force.UAV.Reward)))
ax.legend(loc="best")


# †††††††††††††††††††††††††††††††††††††††柱状图††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
x = 12
fig7, ax7 = plt.subplots(1)
fig7.suptitle('The mean cost of devcies and UAV')
type = ['Random', 'Force', 'Smart']
data111 = [-np.mean(env_random.UAV.Reward), -np.mean(env_force.UAV.Reward),
           -np.mean(logging_timeline[0][x]['UAV_Reward'])]
data1111 = [-np.sum(env_random.UAV.Reward), -np.sum(env_force.UAV.Reward),
            -np.sum(logging_timeline[0][x]['UAV_Reward'])]
data22 = [np.mean(env_random.UAV.Energy), np.mean(env_force.UAV.Energy),
          np.mean(logging_timeline[0][x]['UAV_Energy'])]
data222 = [np.sum(env_random.UAV.Energy), np.sum(env_force.UAV.Energy),
           np.sum(logging_timeline[0][x]['UAV_Energy'])]
ax7.bar(type, [k * param['mu'] for k in data111], label='reward')
ax7.bar(type, [k * param['mu'] for k in data22], bottom=np.array(data111) * param['mu'], label='energy')
ax7.axhline(y=0, color='k', linestyle='-', linewidth='0.6')
ax7.legend(loc="best")
ax7.set_ylabel('Total Cost')  # Add a y-label to the axes.
# ax7.set_title('The Mean')
plt.show()


x = 12
fig8, ax8 = plt.subplots(1)
fig8.suptitle('The mean cost of AoI and CPU')
type = ['Random', 'Force', 'Smart']
dataAoImean = [np.mean(env_random.UAV.AoI), np.mean(env_force.UAV.AoI),
               np.mean(logging_timeline[0][x]['UAV_AoI'])]
dataAoIsum = [np.sum(env_random.UAV.AoI), np.sum(env_force.UAV.AoI),
              np.sum(logging_timeline[0][x]['UAV_AoI'])]
dataCPUmean = [np.mean(env_random.UAV.CPU), np.mean(env_force.UAV.CPU),
               np.mean(logging_timeline[0][x]['UAV_CPU'])]
dataCPUsum = [np.sum(env_random.UAV.CPU), np.sum(env_force.UAV.CPU),
              np.sum(logging_timeline[0][x]['UAV_CPU'])]
ax8.bar(type, [k * param['beta'] for k in dataAoImean], label='AoI')
ax8.bar(type, [k * param['beta'] for k in dataCPUmean], bottom=np.array(dataAoImean) * param['beta'], label='CPU')
ax8.axhline(y=0, color='k', linestyle='-', linewidth='0.6')
ax8.legend(loc="best")
# ax8.set_title('The Mean')
ax8.set_ylabel('Total Cost')  # Add a y-label to the axes.
plt.show()

d = 1



