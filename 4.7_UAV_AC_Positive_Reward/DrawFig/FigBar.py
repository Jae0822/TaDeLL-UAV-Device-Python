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
    pickle.dump([model, env, env_random, env_force, param, avg, logging_timeline], f)



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
logging_timeline[0] is empty. So used it for UAV's data such as:
        logging_timeline[0][x]['UAV_PositionList'] = UAV.PositionList
        logging_timeline[0][x]['UAV_PositionCor'] = UAV.PositionCor
        logging_timeline[0][x]['UAV_Reward'] = UAV.Reward
        logging_timeline[0][x]['UAV_Energy'] = UAV.Energy
"""


# ---------------------------------------------------Fig_temp---------------------------------------------------------------
with open('fig_temp.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)


with open('fig3_5.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)

# ---------------------------------------------------Fig8---------------------------------------------------------------
# with open('fig8.pkl', 'rb') as f:
#     model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)


# ---------------------------------------------------Fig8---------------------------------------------------------------
# with open('fig8.pkl', 'rb') as f:
#     model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)





fig, ax = plt.subplots(1)
# ax[0].plot(np.arange(i_episode), Ep_reward, label='Actor-Critic')
# ax[0].set_xlabel('Episodes')  # Add an x-label to the axes.
# ax[0].set_ylabel('ep_reward')  # Add a y-label to the axes.
# ax[0].set_title("The ep_reward")  # Add a title to the axes.

# ax.plot(np.arange(1, EP+1), Ave_Reward, label='%.0f  Devices, %.0f TimeUnits, %.0f  episodes' %(param['num_Devices'], param['nTimeUnits'], EP, args.gamma, args.learning_rate))
ax.plot(np.arange(1, param['episodes']+1), avg['Ave_Reward'], label= str(param['num_Devices']) + ' Devices,' + str(param['episodes']) + ' episodes,' +  str(param['nTimeUnits']) + ' TimeUnits,' +  str(param['gamma']) + ' gamma,' + str(param['learning_rate']) + ' lr,' + str(param['mu']) + ' mu')
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
# plt.show()

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
# plt.show()
#      https://matplotlib.org/stable/tutorials/text/text_intro.html




fig2, ax2 = plt.subplots(1)
type = ['Random', 'Force', 'Smart']
# data1 = [np.mean([i for i in Reward_random if i >= -30]), np.mean([i for i in Reward_force if i >= -30]), np.mean(logging_timeline[0][param['episodes']]['UAV_Reward'])]
# data2 = [np.mean(PV_random), np.mean(PV_force), np.mean(logging_timeline[0][param['episodes']]['UAV_Energy'])]
# ax2.bar(type, data1, label = 'reward')
# ax2.bar(type, data2, bottom=np.array(data1), label = 'energy')
data11 = [np.mean([i for i in env_random.UAV.Reward if i >= -30]), np.mean([i for i in env_force.UAV.Reward if i >= -30]),
          np.mean(logging_timeline[0][param['episodes']]['UAV_Reward'])]
data22 = [np.mean(env_random.UAV.Energy), np.mean(env_force.UAV.Energy),
          np.mean(logging_timeline[0][param['episodes']]['UAV_Energy'])]
ax2.bar(type, data11, label='reward')
ax2.bar(type, data22, bottom=np.array(data11), label='energy')
ax2.legend(loc="best")
# ax2.set_xlabel('Different Types')  # Add an x-label to the axes.
ax2.set_ylabel('Total Cost')  # Add a y-label to the axes.
# plt.show()
# https: // www.zhihu.com / question / 507977476 / answer / 2283372858
# https: // juejin.cn / post / 6844903664780328974


fig3, ax3 = plt.subplots(1)
type = ['Random', 'Force', 'Smart']
# data1 = [np.mean([i for i in Reward_random if i >= -30]), np.mean([i for i in Reward_force if i >= -30]), np.mean(logging_timeline[0][param['episodes']]['UAV_Reward'])]
# data2 = [np.mean(PV_random), np.mean(PV_force), np.mean(logging_timeline[0][param['episodes']]['UAV_Energy'])]
# ax2.bar(type, data1, label = 'reward')
# ax2.bar(type, data2, bottom=np.array(data1), label = 'energy')
data111 = [np.mean(env_random.UAV.Reward), np.mean(env_force.UAV.Reward),
          np.mean(logging_timeline[0][param['episodes']]['UAV_Reward'])]
data222 = [np.mean(env_random.UAV.Energy), np.mean(env_force.UAV.Energy),
          np.mean(logging_timeline[0][param['episodes']]['UAV_Energy'])]
ax3.axhline(y = 0, color='k', linestyle='-', linewidth='0.6')
ax3.bar(type, data111, label='reward')
ax3.bar(type, data222, bottom = data11, label='energy')
ax3.legend(loc="best")
# ax2.set_xlabel('Different Types')  # Add an x-label to the axes.
ax3.set_ylabel('Total Cost')  # Add a y-label to the axes.
plt.show()