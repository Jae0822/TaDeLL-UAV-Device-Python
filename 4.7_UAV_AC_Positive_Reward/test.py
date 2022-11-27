import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d

with open('fig_A6.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)




Devices = env.Devices


fig, ax = plt.subplots(1)
# for D in Devices:
#     plt.scatter(D.location[0], D.location[1])
[plt.scatter(D.location[0], D.location[1]) for D in Devices]
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.grid(True)



fig, ax = plt.subplots(1)
# ax.plot(np.arange(1, EP+1), Ave_Reward, label='%.0f  Devices, %.0f TimeUnits, %.0f  episodes' %(param['num_Devices'], param['nTimeUnits'], EP, args.gamma, args.learning_rate))
ax.plot(np.arange(1, param['episodes'] + 1), avg['Ep_reward'],
        label=str(param['num_Devices']) + ' Devices,' + str(param['episodes']) + ' episodes,' + str(
            param['nTimeUnits']) + ' TimeUnits,' + str(param['gamma']) + ' gamma,' + str(
            param['learning_rate']) + ' lr,' + str(param['alpha']) + ' alpha, ' + str(param['mu']) + ' mu')
ax.set_xlabel('Episodes')  # Add an x-label to the axes.
ax.set_ylabel('Ave_Reward')  # Add a y-label to the axes.
ax.set_title("The Ave_Reward, NN:" + str(model.pattern))  # Add a title to the axes.
# ax.axhline(y=max(avg['Ave_Reward']), color='r', linestyle='--', linewidth='0.9',
#            label='Smart: ' + str(max(avg['Ave_Reward'])))
# ax.axhline(y=avg['ave_Reward_random'] * len(env_random.UAV.Reward), color='b', linestyle='--', linewidth='0.9',
#            label='Random:' + str(avg['ave_Reward_random'] * len(env_random.UAV.Reward)))
# ax.axhline(y=avg['ave_Reward_force'] * len(env_force.UAV.Reward), color='g', linestyle='--', linewidth='0.9',
#            label='Forced:' + str(avg['ave_Reward_force'] * len(env_force.UAV.Reward)))
ax.legend(loc="best")





a = []
for x in range(1, param['episodes'] + 1):
    print(x)
    a.append(len(logging_timeline[0][x]['UAV_Reward']))
dd = [x*y for x, y in zip(a, avg['Ep_reward'])]
fig, ax = plt.subplots(1)
# ax.plot(np.arange(1, EP+1), Ave_Reward, label='%.0f  Devices, %.0f TimeUnits, %.0f  episodes' %(param['num_Devices'], param['nTimeUnits'], EP, args.gamma, args.learning_rate))
ax.plot(np.arange(1, param['episodes'] + 1), dd,
        label=str(param['num_Devices']) + ' Devices,' + str(param['episodes']) + ' episodes,' + str(
            param['nTimeUnits']) + ' TimeUnits,' + str(param['gamma']) + ' gamma,' + str(
            param['learning_rate']) + ' lr,' + str(param['alpha']) + ' alpha, ' + str(param['mu']) + ' mu')
ax.set_xlabel('Episodes')  # Add an x-label to the axes.
ax.set_ylabel('Ave_Reward')  # Add a y-label to the axes.
ax.set_title("The Ave_Reward, NN:" + str(model.pattern))  # Add a title to the axes.
# ax.axhline(y=max(avg['Ave_Reward']), color='r', linestyle='--', linewidth='0.9',
#            label='Smart: ' + str(max(avg['Ave_Reward'])))
# ax.axhline(y=avg['ave_Reward_random'], color='b', linestyle='--', linewidth='0.9',
#            label='Random:' + str(avg['ave_Reward_random']))
# ax.axhline(y=avg['ave_Reward_force'], color='g', linestyle='--', linewidth='0.9',
#            label='Forced:' + str(avg['ave_Reward_force']))
ax.legend(loc="best")