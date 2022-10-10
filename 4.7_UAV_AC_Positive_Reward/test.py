import numpy as np
import pickle
import matplotlib.pyplot as plt

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

# ---------------------------------------------------Fig7---------------------------------------------------------------
# with open('fig7.pkl', 'rb') as f:
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
