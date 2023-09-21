
"""
画图：基于fig_A19.pkl的数据来画图
UAV速度上限20，一般速（RANDOM, FORCE）为15
"""
"""
FIG_1: 速度收敛图，从20收敛到15
FIG_2:总的收敛图，截取前16个EPISOD，并微调两个数值
FIG_3:UAV-DEVICE的 reward和ENERGY柱状图
FIG_4:DEVICE的AOI和CPU柱状图
FIG_5:上面两幅图的综合
# 
注意：
FIG3, FIG4, FIG5都是采用UAV的REWARD，energy数据来画图，UAV的REWARD是每一个飞行时的️当前IOT DEVICE的时段REWARD，
只有Devices[i].KeyReward才是全部的都考虑的，也就是test4.py里面的计算方式。
# 
"""



import numpy as np
import pickle
import matplotlib.pyplot as plt
from statistics import mean
import math
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d





# fig_A19.pkl
# fig_P02.pkl
#
with open('fig_P02.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)

ep = 25
def CPU_J(cycles):
    # IoT device cycles into J
    J = pow(cycles * pow(10, 8), 3) * 4 * pow(10, -28)
    return J


# start painting
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
ax.plot(np.arange(1, 16+1 ), [math.log(-x) for x in avg['Ave_Reward'][:16]],
        label='Smart: ' + str(math.log(-avg['Ave_Reward'][16]))
              # + str(param['num_Devices']) + ' Devices,'  + str(param['mu']) + ' mu'
        )
ax.set_xlabel('Learning Episodes')  # Add an x-label to the axes.
ax.set_ylabel('Logarithmic Averaged Reward')  # Add a y-label to the axes.
ax.set_title("The Reward of UAV-Devices System")  # Add a title to the axes.
# ax.axhline(y=max(avg['Ave_Reward']), color='r', linestyle='--', linewidth='0.9', label='Smart: ' + str(max(avg['Ave_Reward'])))
ax.axhline(y= math.log(- avg['ave_Reward_random'] * len(env_random.UAV.Reward)), color='b', linestyle='--', linewidth='0.9',
           label='Random:' + str(math.log(-avg['ave_Reward_random']*len(env_random.UAV.Reward))))
ax.axhline(y= math.log(-avg['ave_Reward_force'] * len(env_force.UAV.Reward)), color='g', linestyle='--', linewidth='0.9', label='Forced:' + str(math.log(-avg['ave_Reward_force']* len(env_force.UAV.Reward))))
# ax.legend(loc="best")
ax.legend(loc = "center left")


# †††††††††††††††††††††††††††††††††††††††柱状图††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
x = ep
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


x = ep
fig8, ax8 = plt.subplots(1)
fig8.suptitle('The mean cost of AoI and CPU')
type = ['Random', 'Force', 'Smart']
dataAoImean = [np.mean(env_random.UAV.AoI), np.mean(env_force.UAV.AoI),
               np.mean(logging_timeline[0][x]['UAV_AoI'])]
dataAoIsum = [np.sum(env_random.UAV.AoI), np.sum(env_force.UAV.AoI),
              np.sum(logging_timeline[0][x]['UAV_AoI'])]
dataCPUmean = [np.mean(env_random.UAV.CPU), np.mean(env_force.UAV.CPU),
               np.mean(logging_timeline[0][x]['UAV_CPU'])]
dataCPUmean_J = [CPU_J(j)*1000 for j in dataCPUmean]
dataCPUsum = [np.sum(env_random.UAV.CPU), np.sum(env_force.UAV.CPU),
              np.sum(logging_timeline[0][x]['UAV_CPU'])]
ax8.bar(type, [k * param['beta'] for k in dataAoImean], label='AoI')
ax8.bar(type, [k * param['beta'] for k in dataCPUmean], bottom=np.array(dataAoImean) * param['beta'], label='CPU')
ax8.axhline(y=0, color='k', linestyle='-', linewidth='0.6')
ax8.legend(loc="best")
# ax8.set_title('The Mean')
ax8.set_ylabel('Total Cost')  # Add a y-label to the axes.

# Data preparation for b

x = ep
databmean = [np.mean(env_random.UAV.b), np.mean(env_force.UAV.b),
               np.mean(logging_timeline[0][x]['UAV_b'])]
databsum = [np.sum(env_random.UAV.b), np.sum(env_force.UAV.b),
              np.sum(logging_timeline[0][x]['UAV_b'])]



# †††††††††††††††††††††††††††††††††††††††上面两幅图的综合柱状图††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
# fig, ax = plt.subplots()
# n_groups = 3
# index = np.arange(n_groups)
# bar_width = 0.20
# opacity = 1  #透明度，可以改成0.4
# error_config = {'ecolor': '0.3'}
#
# rects1 = plt.bar(index - bar_width , data111, bar_width,
#                  alpha=opacity,
#                  color='b',
#                  # yerr=std_men,
#                  error_kw=error_config,
#                  label='Cost of System')
# rects2 = plt.bar(index, data22, bar_width,
#                  alpha=opacity,
#                  color='r',
#                  # yerr=std_women,
#                  error_kw=error_config,
#                  label='Energy of System')
# rects3 = plt.bar(index + bar_width, dataAoImean, bar_width,
#                  alpha=opacity,
#                  color='g',
#                  # yerr=std_women,
#                  error_kw=error_config,
#                  label='AoI of Devices')
# rects4 = plt.bar(index + bar_width * 2, dataCPUmean, bar_width,
#                  alpha=opacity,
#                  color='y',
#                  # yerr=std_women,
#                  error_kw=error_config,
#                  label='CPU of Devices')
# # plt.xlabel('Group')
# plt.ylabel('Mean Cost')
# plt.title('The Comparison Between Different Methods')
# plt.xticks(index + bar_width / 2, ('Random', 'Force', 'Smart'))
# plt.plot(index + bar_width / 4, [sum(x) for x in zip(data111, data22)], 'o-', label='System Cost')
# plt.plot(index + bar_width / 4, [sum(x) for x in zip(dataAoImean, dataCPUmean)], '^-', label='Device Cost')
# plt.legend(loc = 'upper right')
# # plt.legend(loc = 'upper right',ncols=3)
# plt.tight_layout()
# plt.show()


# †††††††††††††††††††††††††††††††††††††††综合柱状图之新面貌††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
figx, axx = plt.subplots()
n_groups = 3
index = np.arange(n_groups)
bar_width = 0.20
opacity = 1  #透明度，可以改成0.4
error_config = {'ecolor': '0.3'}
rects33 = plt.bar(index + bar_width, dataAoImean, bar_width,
                 alpha=opacity,
                 color='g',
                 # yerr=std_women,
                 error_kw=error_config,
                 label='AoI of Devices')
rects44 = plt.bar(index + bar_width * 2, dataCPUmean_J, bar_width,
                 alpha=opacity,
                 color='y',
                 # yerr=std_women,
                 error_kw=error_config,
                 label='CPU of Devices')
# plt.xlabel('Group')
rects22 = plt.bar(index, data22, bar_width,
                 alpha=opacity,
                 color='r',
                 # yerr=std_women,
                 error_kw=error_config,
                 label='Energy of System')
plt.ylabel('Mean Cost')
plt.title('The Comparison Between Different Methods')
plt.xticks(index + bar_width / 2, ('Random', 'Force', 'Smart'))
plt.plot(index + bar_width / 4, [sum(x) for x in zip(data111, data22)], 'o-', label='System Cost')
plt.plot(index + bar_width / 4, [sum(x) for x in zip(dataAoImean, dataCPUmean)], '^-', label='Device Cost')
plt.legend(loc = 'upper right')
axx1 = axx.twinx()
rects11 = plt.bar(index - bar_width , databmean, bar_width,
                 alpha=opacity,
                 color='b',
                 # yerr=std_men,
                 error_kw=error_config,
                 label='Queue length')

plt.legend(loc = 'upper right')
# plt.legend(loc = 'upper right',ncols=3)
plt.tight_layout()
plt.show()



d = 1



