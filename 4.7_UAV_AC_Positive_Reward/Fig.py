import numpy as np
import pickle
import matplotlib.pyplot as plt
# https://www.journaldev.com/15638/python-pickle-example

"""
uncomment对应文件，画相应的图
下面是数据格式
"""

"""
用这个命令储存数据：
with open('fig_temp.pkl', 'wb') as f:
    pickle.dump([model, env, param, avg, logging_timeline], f)

1)model
model = Policy(1 * param['num_Devices'] + 2, param['num_Devices']) # 待定不一定有+2
2)env
env = Env(Devices, UAV, param['nTimeUnits'])
3)param = {'episodes': 2, 'nTimeUnits': 5, 'nTimeUnits_random': 2, 'nTimeUnits_force': 2,
         'gamma': 0.6, 'learning_rate': 3e-1, 'log_interval': 1, 'seed': 0,
         'num_Devices': 9, 'V': 72, 'field': 1, 'dist': 0.040, 'freq_low': 8, 'freq_high': 16}
4)avg = {}
avg['Ave_Reward'] = Ave_Reward
avg['ave_Reward_random'] = ave_Reward_random
avg['ave_Reward_force'] = ave_Reward_force
5)logging for each episode:
logging_timeline = [ device0, device1, device2....,  ,  ]
device = [episode0, episode1, episode2, ...,  ]
episode = {'intervals': [], 'rewards': []}
"""


# ---------------------------------------------------Fig_temp---------------------------------------------------------------
with open('fig_temp.pkl', 'rb') as f:
    model, env, param, avg, logging_timeline = pickle.load(f)


# ---------------------------------------------------Fig1---------------------------------------------------------------
# with open('fig1.pkl', 'rb') as f:
#     model, param, avg, logging_timeline = pickle.load(f)
# avg['Ave_Reward'][1] = -4.700195220179078
# avg['Ave_Reward'][4] = -4.588885932705906

# ---------------------------------------------------Fig2---------------------------------------------------------------
# with open('fig2.pkl', 'rb') as f:
#     model, param, avg, logging_timeline = pickle.load(f)


# ---------------------------------------------------Fig3---------------------------------------------------------------
# with open('fig3.pkl', 'rb') as f:
#     model, param, avg, logging_timeline = pickle.load(f)

# ---------------------------------------------------Fig4---------------------------------------------------------------
# with open('fig4.pkl', 'rb') as f:
#     model, param, avg, logging_timeline = pickle.load(f)

# ---------------------------------------------------Fig5---------------------------------------------------------------
# with open('fig5.pkl', 'rb') as f:
#     model, param, avg, logging_timeline = pickle.load(f)


# ---------------------------------------------------Fig6---------------------------------------------------------------
# with open('fig6.pkl', 'rb') as f:
#     model, param, avg, logging_timeline = pickle.load(f)




fig, ax = plt.subplots(1)
# ax[0].plot(np.arange(i_episode), Ep_reward, label='Actor-Critic')
# ax[0].set_xlabel('Episodes')  # Add an x-label to the axes.
# ax[0].set_ylabel('ep_reward')  # Add a y-label to the axes.
# ax[0].set_title("The ep_reward")  # Add a title to the axes.

# ax.plot(np.arange(1, EP+1), Ave_Reward, label='%.0f  Devices, %.0f TimeUnits, %.0f  episodes' %(param['num_Devices'], param['nTimeUnits'], EP, args.gamma, args.learning_rate))
ax.plot(np.arange(1, param['episodes']+1), avg['Ave_Reward'], label= str(param['num_Devices']) + ' Devices,' + str(param['episodes']) + ' episodes,' +  str(param['nTimeUnits']) + ' TimeUnits,' +  str(param['gamma']) + ' gamma,' + str(param['learning_rate']) + ' lr')
ax.set_xlabel('Episodes')  # Add an x-label to the axes.
ax.set_ylabel('Ave_Reward')  # Add a y-label to the axes.
ax.set_title("The Ave_Reward, NN:" + str(model.pattern))  # Add a title to the axes.
ax.axhline(y = max(avg['Ave_Reward']), color='r', linestyle='--', linewidth='0.9',   label='Smart: ' + str(max(avg['Ave_Reward'])))
ax.axhline(y = avg['ave_Reward_random'], color='b', linestyle='--', linewidth='0.9', label='Random:'+ str(avg['ave_Reward_random']))
ax.axhline(y = avg['ave_Reward_force'], color='g', linestyle='--', linewidth='0.9', label='Forced:'+ str(avg['ave_Reward_force']))
ax.legend(loc="best")


# ax[1].plot(np.arange(i_episode), [ave_Reward_random]*i_episode, label='Random')
# ax[1].set_xlabel('Episodes')  # Add an x-label to the axes.
# ax[1].set_ylabel('Ave_Reward')  # Add a y-label to the axes.
# ax[1].set_title("The Ave_Reward")  # Add a title to the axes.
# plt.legend()
plt.show()

x = 1

fig1, ax1 = plt.subplots(param['num_Devices'])
fig1.supxlabel('Time Unites for one episode')
fig1.supylabel('The Ave Reward')
fig1.suptitle('The episode %.0f' %(x))
for i in range(param['num_Devices']):
    if len(logging_timeline[i][x]['intervals']) == 0:
        ax1[i].text(0.4, 0.5, 'No Visit by UAV')
    else:
        ax1[i].step(logging_timeline[i][x]['KeyTime'], logging_timeline[i][x]['KeyRewards'], '^-g', where='post',
                    label='device %.0f' % (i))
        ax1[i].set_xlim([0, param['nTimeUnits']])
        ax1[i].axhline(y=logging_timeline[i][x]['avg_reward'], color='r', linestyle='--', linewidth='0.9',
                       label=logging_timeline[i][x]['avg_reward'])
        ax1[i].legend(loc="best")
        # ax1[i].text(2, logging_timeline[i][x]['avg_reward'], logging_timeline[i][x]['avg_reward'], verticalalignment='bottom',horizontalalignment='left', rotation=360, color='r')
        ax1[i].set_ylabel('device  %.0f' % (i))
        # if i == 0:
        #     ax1[i].set_title(model.pattern)
        # ax1[i].set_title('CPU Capacity: %.0f' % (Devices[i].cpu_capacity))
        for vv in range(len(np.where(logging_timeline[i][x]['TimeList'])[0])):
            ax1[i].axvline(x=np.where(logging_timeline[i][x]['TimeList'])[0][vv], linestyle='--', linewidth='0.9')
            # ax1[i].plot([np.where(Devices[i].TimeList)],[logging_timeline[i][x]['rewards']], 'o')
plt.show()
#      https://matplotlib.org/stable/tutorials/text/text_intro.html

