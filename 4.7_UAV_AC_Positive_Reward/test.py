import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d

with open('fig_A6.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)





Devices = env.Devices

dis = []
for i in range(len(Devices)):
    d = []
    for j in range(len(Devices)):
        distance = np.linalg.norm(Devices[i].location - Devices[j].location)  # Compute the distance of two points
        d.append(distance)
    dis.append(d)





# †††††††††††††††††††††††††††††††††††††††EP_REWARD††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
fig_ep, ax_ep = plt.subplots(1)
# ax.plot(np.arange(1, EP+1), Ave_Reward, label='%.0f  Devices, %.0f TimeUnits, %.0f  episodes' %(param['num_Devices'], param['nTimeUnits'], EP, args.gamma, args.learning_rate))
ax_ep.plot(np.arange(1, param['episodes'] + 1), avg['Ep_reward'],
        label=str(param['num_Devices']) + ' Devices,' + str(param['episodes']) + ' episodes,' + str(
            param['nTimeUnits']) + ' TimeUnits,' + str(param['gamma']) + ' gamma,' + str(
            param['learning_rate']) + ' lr,' + str(param['alpha']) + ' alpha, ' + str(param['mu']) + ' mu')
ax_ep.set_xlabel('Episodes')  # Add an x-label to the axes.
ax_ep.set_ylabel('Ave_Reward')  # Add a y-label to the axes.
ax_ep.set_title("The reward divided by number of flights, NN:" + str(model.pattern))  # Add a title to the axes.
# ax.axhline(y=max(avg['Ave_Reward']), color='r', linestyle='--', linewidth='0.9',
#            label='Smart: ' + str(max(avg['Ave_Reward'])))
ax_ep.axhline(y=avg['ave_Reward_random'], color='b', linestyle='--', linewidth='0.9',
           label='Random:' + str(avg['ave_Reward_random']))
ax_ep.axhline(y=avg['ave_Reward_force'], color='g', linestyle='--', linewidth='0.9',
           label='Forced:' + str(avg['ave_Reward_force']))
ax_ep.legend(loc="best")



# EP_reward复现SUMMATION
a = []
for x in range(1, param['episodes'] + 1):
    # print(x)
    a.append(len(logging_timeline[0][x]['UAV_Reward']))
fig, ax = plt.subplots(1)
# ax.plot(np.arange(1, EP+1), Ave_Reward, label='%.0f  Devices, %.0f TimeUnits, %.0f  episodes' %(param['num_Devices'], param['nTimeUnits'], EP, args.gamma, args.learning_rate))
ax.plot(np.arange(1, param['episodes'] + 1), [x*y for x, y in zip(a, avg['Ep_reward'])],
        label=str(param['num_Devices']) + ' Devices,' + str(param['episodes']) + ' episodes,' + str(
            param['nTimeUnits']) + ' TimeUnits,' + str(param['gamma']) + ' gamma,' + str(
            param['learning_rate']) + ' lr,' + str(param['alpha']) + ' alpha, ' + str(param['mu']) + ' mu')
ax.set_xlabel('Episodes')  # Add an x-label to the axes.
ax.set_ylabel('Ave_Reward')  # Add a y-label to the axes.
ax.set_title("The Ave_Reward, NN:" + str(model.pattern))  # Add a title to the axes.
# ax.axhline(y=max(avg['Ave_Reward']), color='r', linestyle='--', linewidth='0.9',
#            label='Smart: ' + str(max(avg['Ave_Reward'])))
ax.axhline(y=avg['ave_Reward_random'] * len(env_random.UAV.Reward), color='b', linestyle='--', linewidth='0.9',
           label='Random:' + str(avg['ave_Reward_random'] * len(env_random.UAV.Reward)))
ax.axhline(y=avg['ave_Reward_force'] * len(env_force.UAV.Reward), color='g', linestyle='--', linewidth='0.9',
           label='Forced:' + str(avg['ave_Reward_force'] * len(env_force.UAV.Reward)))
ax.legend(loc="best")



x = 15
fig5, ax5 = plt.subplots(1)
type = ['Random', 'Force', 'Smart']
# data1 = [np.mean([i for i in Reward_random if i >= -30]), np.mean([i for i in Reward_force if i >= -30]), np.mean(logging_timeline[0][param['episodes']]['UAV_Reward'])]
# data2 = [np.mean(PV_random), np.mean(PV_force), np.mean(logging_timeline[0][param['episodes']]['UAV_Energy'])]
# ax2.bar(type, data1, label = 'reward')
# ax2.bar(type, data2, bottom=np.array(data1), label = 'energy')
# data11 = [- np.mean([i for i in env_random.UAV.Reward if i >= -30]),
#           -np.mean([i for i in env_force.UAV.Reward if i >= -30]),
#           -np.mean(logging_timeline[0][param['episodes']]['UAV_Reward'])]
data111 = [-np.mean(env_random.UAV.Reward), -np.mean(env_force.UAV.Reward),
           -np.mean(logging_timeline[0][x]['UAV_Reward'])]
data1111 = [-np.sum(env_random.UAV.Reward), -np.sum(env_force.UAV.Reward),
            -np.sum(logging_timeline[0][x]['UAV_Reward'])]
data22 = [np.mean(env_random.UAV.Energy), np.mean(env_force.UAV.Energy),
          np.mean(logging_timeline[0][x]['UAV_Energy'])]
data222 = [np.sum(env_random.UAV.Energy), np.sum(env_force.UAV.Energy),
           np.sum(logging_timeline[0][x]['UAV_Energy'])]
ax5.bar(type, [k * param['mu'] for k in data111], label='reward')
ax5.bar(type, [k * param['mu'] for k in data22], bottom=np.array(data111) * param['mu'], label='energy')
ax5.axhline(y=0, color='k', linestyle='-', linewidth='0.6')
ax5.legend(loc="best")
fig5.suptitle('The Mean')
# ax2.set_xlabel('Different Types')  # Add an x-label to the axes.
ax5.set_ylabel('Total Cost')  # Add a y-label to the axes.
plt.show()

fig2, ax2 = plt.subplots(1)
type = ['Random', 'Force', 'Smart']
# data1 = [np.mean([i for i in Reward_random if i >= -30]), np.mean([i for i in Reward_force if i >= -30]), np.mean(logging_timeline[0][param['episodes']]['UAV_Reward'])]
# data2 = [np.mean(PV_random), np.mean(PV_force), np.mean(logging_timeline[0][param['episodes']]['UAV_Energy'])]
# ax2.bar(type, data1, label = 'reward')
# ax2.bar(type, data2, bottom=np.array(data1), label = 'energy')
# data11 = [- np.mean([i for i in env_random.UAV.Reward if i >= -30]),
#           -np.mean([i for i in env_force.UAV.Reward if i >= -30]),
#           -np.mean(logging_timeline[0][param['episodes']]['UAV_Reward'])]
data111 = [-np.mean(env_random.UAV.Reward), -np.mean(env_force.UAV.Reward),
           -np.mean(logging_timeline[0][x]['UAV_Reward'])]
data1111 = [-np.sum(env_random.UAV.Reward), -np.sum(env_force.UAV.Reward),
            -np.sum(logging_timeline[0][x]['UAV_Reward'])]
data22 = [np.mean(env_random.UAV.Energy), np.mean(env_force.UAV.Energy),
          np.mean(logging_timeline[0][x]['UAV_Energy'])]
data222 = [np.sum(env_random.UAV.Energy), np.sum(env_force.UAV.Energy),
           np.sum(logging_timeline[0][x]['UAV_Energy'])]
# [a + b for (a, b) in zip(data111, data22)]
# [a + b for (a, b) in zip(data1111, data222)]
ax2.bar(type, [k * param['mu'] for k in data1111], label='reward')
ax2.bar(type, [k * param['mu'] for k in data222], bottom=np.array(data1111) * param['mu'], label='energy')
ax2.axhline(y=0, color='k', linestyle='-', linewidth='0.6')
ax2.legend(loc="best")
fig2.suptitle('The Sum')
# ax2.set_xlabel('Different Types')  # Add an x-label to the axes.
ax2.set_ylabel('Total Cost')  # Add a y-label to the axes.
plt.show()