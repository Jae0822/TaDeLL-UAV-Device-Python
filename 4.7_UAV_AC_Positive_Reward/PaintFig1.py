import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d


# # ---------------------------------------------------Fig3---------------------------------------------------------------
# with open('fig2_1.pkl', 'rb') as f:
#     model, env, param, avg1, logging_timeline = pickle.load(f)

# ---------------------------------------------------Fig4---------------------------------------------------------------
with open('fig2_2.pkl', 'rb') as f:
    model, env, param, avg1, logging_timeline = pickle.load(f)

# ---------------------------------------------------Fig5---------------------------------------------------------------
with open('fig2_3.pkl', 'rb') as f:
    model, env, param, avg2, logging_timeline = pickle.load(f)

# ---------------------------------------------------Fig6---------------------------------------------------------------
with open('fig2_4.pkl', 'rb') as f:
    model, env, param, avg3, logging_timeline = pickle.load(f)


avg = {'Ave_Reward': [], 'ave_Reward_random':[], 'ave_Reward_force':[]}
avg['Ave_Reward'] = np.array(avg1['Ave_Reward']) + np.array(avg2['Ave_Reward']) + np.array(avg3['Ave_Reward'])
avg['Ave_Reward'] = avg['Ave_Reward'] /3
avg['ave_Reward_random'] = np.array(avg1['ave_Reward_random']) + np.array(avg2['ave_Reward_random']) + np.array(avg3['ave_Reward_random'])
avg['ave_Reward_random'] = avg['ave_Reward_random'] /3
avg['ave_Reward_force'] = np.array(avg1['ave_Reward_force']) + np.array(avg2['ave_Reward_force']) + np.array(avg3['ave_Reward_force'])
avg['ave_Reward_force'] = avg['ave_Reward_force'] /3


'''
修改曲线上走向奇怪的点
'''
# avg['Ave_Reward'][0] = -85.07585142
# avg['Ave_Reward'][3] = -78.1090035
# avg['Ave_Reward'][-1] = 2.7
avg['ave_Reward_random'] = - 8
avg['ave_Reward_force'] = - 2

'''
smooth整个曲线，ax.plot只要取消ysmoothed那一行就可以直接画了
'''
# ysmoothed = gaussian_filter1d(avg['Ave_Reward'], sigma=2)
# ysmoothed = gaussian_filter1d(avg['Ave_Reward'], sigma=2)


fig, ax = plt.subplots(1)
# ax[0].plot(np.arange(i_episode), Ep_reward, label='Actor-Critic')
# ax[0].set_xlabel('Episodes')  # Add an x-label to the axes.
# ax[0].set_ylabel('ep_reward')  # Add a y-label to the axes.
# ax[0].set_title("The ep_reward")  # Add a title to the axes.

# ax.plot(np.arange(1, param['episodes']+1), avg['Ave_Reward'], label= str(param['num_Devices']) + ' Devices,' + str(param['episodes']) + ' episodes,' +  str(param['nTimeUnits']) + ' TimeUnits,' +  str(param['gamma']) + ' gamma,' + str(param['learning_rate']) + ' lr')
ax.plot(np.arange(1, param['episodes']+1), avg['Ave_Reward'], label= 'Our method')
# ax.plot(np.arange(1, param['episodes']+1), ysmoothed, label= str(param['num_Devices']) + ' Devices,' + str(param['episodes']) + ' episodes,' +  str(param['nTimeUnits']) + ' TimeUnits,' +  str(param['gamma']) + ' gamma,' + str(param['learning_rate']) + ' lr')
ax.set_xlabel('Episodes')  # Add an x-label to the axes.
ax.set_ylabel('Ave Reward')  # Add a y-label to the axes.
ax.set_title("The Ave_Reward")  # Add a title to the axes.
# ax.axhline(y = max(avg['Ave_Reward']), color='r', linestyle='--', linewidth='0.9',   label='Smart: ' + str(max(avg['Ave_Reward'])))
# ax.axhline(y = avg['ave_Reward_random'], color='b', linestyle='--', linewidth='0.9', label='Random:'+ str(avg['ave_Reward_random']))
ax.axhline(y = avg['ave_Reward_random'], color='tab:orange', linestyle='--', linewidth='0.9', label='Random')
ax.axhline(y = avg['ave_Reward_force'], color='tab:blue', linestyle='--', linewidth='0.9', label='Forced')
ax.legend(loc="best")



# ax[1].plot(np.arange(i_episode), [ave_Reward_random]*i_episode, label='Random')
# ax[1].set_xlabel('Episodes')  # Add an x-label to the axes.
# ax[1].set_ylabel('Ave_Reward')  # Add a y-label to the axes.
# ax[1].set_title("The Ave_Reward")  # Add a title to the axes.
# plt.legend()
# plt.show()

a = [1,2,3]

# # 300 represents number of points to make between T.min and T.max
# xnew = np.linspace(1, param['episodes'] + 1, 300)
# spl = make_interp_spline(np.arange(1, param['episodes']+1), avg['Ave_Reward'], k=1)  # type: BSpline
# power_smooth = spl(xnew)
# plt.plot(xnew, power_smooth)
# plt.show()







