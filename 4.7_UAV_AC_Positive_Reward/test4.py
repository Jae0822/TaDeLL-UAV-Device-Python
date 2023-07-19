import numpy as np
import pickle
import matplotlib.pyplot as plt
from statistics import mean

"""
画速度收敛图的源代码，和修正版（不采纳）
"""

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

d = 1


V = []
for x in range(1, param['episodes']):
    V = V + logging_timeline[0][x]['UAV_VelocityList'][1]
fig_V, ax_V = plt.subplots(1)
ax_V.plot(np.arange(1, len(V) + 1), V)
# ax_V.plot(np.arange(1, len(logging_timeline[0][x]['UAV_VelocityList']) + 1), logging_timeline[0][x]['UAV_VelocityList'])
ax_V.set_ylabel('Velocity(m/s)')
ax_V.set_xlabel('Episodes')
ax_V.grid(True)

