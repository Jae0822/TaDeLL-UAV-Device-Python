import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d
import math
import random

from env import Task

fig, ax = plt.subplots(1)
fig.suptitle('1')


fig8, ax8 = plt.subplots(2)
ax8[0].plot(range(10), 'r')
ax8[1].plot(range(10), 'r')
fig8.suptitle('8')

fig9, ax9 = plt.subplots(2,2)
ax9[0,0].plot(range(10), 'r')
ax9[0,1].plot(range(10), 'b')
ax9[1,0].plot(range(10), 'g')
ax9[1,1].plot(range(10), 'k')
fig9.suptitle('9')


V_avg = []
for x in range(1, param['episodes']):
    V_avg.append(mean(logging_timeline[0][x]['UAV_VelocityList']))
fig_v, ax_v = plt.subplots(1)
ax_v.plot(np.arange(1, param['episodes']), V_avg)
fig_v.suptitle('Velocity trend')

fig7, ax7 = plt.subplots(2, sharex=True)
fig7.suptitle('Devcie and UAV cost')
type = ['Random', 'Force', 'Smart']
data111 = [-np.mean(env_random.UAV.Reward), -np.mean(env_force.UAV.Reward),
           -np.mean(logging_timeline[0][10]['UAV_Reward'])]
data1111 = [-np.sum(env_random.UAV.Reward), -np.sum(env_force.UAV.Reward),
            -np.sum(logging_timeline[0][10]['UAV_Reward'])]
data22 = [np.mean(env_random.UAV.Energy), np.mean(env_force.UAV.Energy),
          np.mean(logging_timeline[0][10]['UAV_Energy'])]
data222 = [np.sum(env_random.UAV.Energy), np.sum(env_force.UAV.Energy),
           np.sum(logging_timeline[0][10]['UAV_Energy'])]
ax7[0].bar(type, [k * param['mu'] for k in data111], label='reward')
ax7[0].bar(type, [k * param['mu'] for k in data22], bottom=np.array(data111) * param['mu'], label='energy')
ax7[0].axhline(y=0, color='k', linestyle='-', linewidth='0.6')
ax7[0].legend(loc="best")
ax7[0].set_ylabel('Total Cost')  # Add a y-label to the axes.
ax7[0].set_title('The Mean')
ax7[1].bar(type, [k * param['mu'] for k in data1111], label='reward')
ax7[1].bar(type, [k * param['mu'] for k in data222], bottom=np.array(data1111) * param['mu'], label='energy')
ax7[1].axhline(y=0, color='k', linestyle='-', linewidth='0.6')
ax7[1].legend(loc="best")
ax7[1].set_ylabel('Total Cost')  # Add a y-label to the axes.
ax7[1].set_title('The Sum')
plt.show()

fig8, ax8 = plt.subplots(2, sharex=True)
fig8.suptitle('AoI and CPU cost')
type = ['Random', 'Force', 'Smart']
dataAoImean = [np.mean(env_random.UAV.AoI), np.mean(env_force.UAV.AoI),
               np.mean(logging_timeline[0][10]['UAV_AoI'])]
dataAoIsum = [np.sum(env_random.UAV.AoI), np.sum(env_force.UAV.AoI),
              np.sum(logging_timeline[0][10]['UAV_AoI'])]
dataCPUmean = [np.mean(env_random.UAV.CPU), np.mean(env_force.UAV.CPU),
               np.mean(logging_timeline[0][10]['UAV_CPU'])]
dataCPUsum = [np.sum(env_random.UAV.CPU), np.sum(env_force.UAV.CPU),
              np.sum(logging_timeline[0][10]['UAV_CPU'])]
ax8[0].bar(type, [k * param['beta'] for k in dataAoImean], label='AoI')
ax8[0].bar(type, [k * param['beta'] for k in dataCPUmean], bottom=np.array(dataAoImean) * param['beta'], label='CPU')
ax8[0].axhline(y=0, color='k', linestyle='-', linewidth='0.6')
ax8[0].legend(loc="best")
ax8[0].set_title('The Mean')
ax8[0].set_ylabel('The mean cost')  # Add a y-label to the axes.
ax8[1].bar(type, [k * param['beta'] for k in dataAoIsum], label='AoI')
ax8[1].bar(type, [k * param['beta'] for k in dataCPUsum], bottom=np.array(dataAoIsum) * param['beta'], label='CPU')
ax8[1].axhline(y=0, color='k', linestyle='-', linewidth='0.6')
ax8[1].legend(loc="best")
ax8[1].set_title('The Sum')
ax8[1].set_ylabel('The sum Cost')  # Add a y-label to the axes.
plt.show()









