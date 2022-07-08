import argparse
import numpy as np
import random
from itertools import count
from collections import namedtuple
import copy
import matplotlib.pyplot as plt
# import matplotlib
from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# import pdb

from IoTEnv import Uav, Device, Env, Policy
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# Press ⌘F8 to toggle the breakpoint.


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--learning_rate', type=float, default=3e-1, metavar='lr',
                    help='discount factor (default: 3e-1)')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

np.random.seed(args.seed)
torch.manual_seed(args.seed)



# Prepare the environment and devices
#  V: 72 km/h =  20 m/s
#  field: 1 km * 1km
#  dist:
param = {'episodes': 2,'nTimeUnits': 2, 'num_Devices': 5, 'V': 72, 'field': 1, 'dist': 0.040, 'freq_low': 8, 'freq_high': 16}
Devices = []
# for i in range(param['num_Devices']):
#     Devices.append(Device(random.randint(param['freq_low'], param['freq_high']), random.randint(30, 70), param['field']))
# Devices.append(Device(35, 30, param['field']))
# Devices.append(Device(35, 40, param['field']))
# Devices.append(Device(35, 50, param['field']))
# Devices.append(Device(35, 60, param['field']))
# Devices.append(Device(35, 70, param['field']))
# Devices.append(Device(50, random.randint(30, 70), param['field']))
# Devices.append(Device(40, random.randint(30, 70), param['field']))
# Devices.append(Device(30, random.randint(30, 70), param['field']))
Devices.append(Device(20, random.randint(30, 70), param['field']))
Devices.append(Device(12, random.randint(30, 70), param['field']))
# Devices.append(Device(9, random.randint(30, 70), param['field']))
Devices.append(Device(40, random.randint(30, 70), param['field']))
Devices.append(Device(5, random.randint(30, 70), param['field']))
Devices.append(Device(4, random.randint(30, 70), param['field']))
# Devices.append(Device(2, random.randint(30, 70), param['field']))


UAV = Uav(param['V'])
env = Env(Devices, UAV, param['nTimeUnits'])

Devices_random = copy.deepcopy(Devices)
UAV_random = copy.deepcopy(UAV)
env_random = Env(Devices_random, UAV_random, param['nTimeUnits'])

model = Policy(1 * param['num_Devices'] + 2, param['num_Devices'])
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)  # lr=3e-2
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    # state = torch.from_numpy(state).float()
    state = torch.from_numpy(state).double()
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()
    # action = torch.tensor(2)


    # print(state)
    for label, p in enumerate(probs):
        print(f'{label:2}: {100 * p:5.2f}%')
    # print("---", action, "is chosen")

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take
    return action.item()

def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []  # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]

def learning():

    rate1 = []
    rate2 = []

    env.initialization(Devices, UAV)

    for i_episode in count(1):

        state = env.reset(Devices, UAV)
        # print("the initial state: ", state)
        print("----------------------------------------------------------------------------")
        print("       ")

        model.states.append(state)
        ep_reward = 0
        t = 0
        n = 0  # logging fly behaviors
        # FIXME: when the distance is large or the velocity is small, the Fly_time can be too large to surpass the nTimeUnits


        rate1.append([])
        rate2.append([])
        while t < param['nTimeUnits']:
        # for t in range(0, param['nTimeUnits']):

            # select action from policy
            action = select_action(state)
            # random action
            # action = np.random.randint(param['num_Devices'])

            # take the action
            # state, reward, reward_Regular, t = env.step(state, action, t)
            t = t + 1
            state, reward_, reward_rest, reward = env.step(state, action, t)
            print(reward_)
            print(reward_rest)
            print(reward)
            n += 1

            rate1[-1].append(reward_ / reward)
            rate2[-1].append(reward_rest / reward)

            # 强制action
            # 强制选择action
            # action_table = np.zeros(param['num_Devices'])  # 筛选出当前有新任务的设备
            # for i in range(param['num_Devices']):
            #     if Devices[i].TimeList[t] == 1:
            #         action_table[i] = 1
            # inx = np.where(action_table == 1)[0]
            # # action = inx[np.random.randint(len(inx))] if inx else np.random.randint(param['num_Devices']) # 随机选一个去访问
            # if inx.any():
            #     action = inx[0]  # inx[np.random.randint(len(inx))] #优先选择变化最不频繁的
            # else:
            #     action = np.random.randint(param['num_Devices'])



            # print("the action          ", action)
            # print("the state:          ", state)
            # print("the reward_         ", reward_)  # current device
            # print("the rest reward:    ", reward_rest)  # of other devices
            # print("the sum reward:     ", reward)  # reward_ + reward_rest

            model.actions.append(action)
            model.states.append(state)


            # model.reward_.append(reward_)
            # model.reward_rest.append(reward_rest)
            model.rewards.append(reward)
            ep_reward += reward

            print("Forced: The {} episode" " and the {} fly" " at the end of {} time slots. " "Visit device {}".format(i_episode, n, t, action))

            print("----------------------------------------------------------------------------")
            print("       ")

        #  Average Reward
        ave_Reward =ep_reward / param['nTimeUnits']

        # update cumulative reward
        # running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        Ep_reward.append(ep_reward)
        # Running_reward.append(running_reward)
        Ave_Reward.append(ave_Reward)

        # for i in range(param['num_Devices']):
        #     logging_timeline[i][EP]['timeline'].append(logging_timeline[i][EP]['intervals'][0])
        #     for j in range(1, len(logging_timeline[i][EP]['intervals'])):
        #         logging_timeline[i][EP]['timeline'].append(logging_timeline[i][EP]['timeline'][j-1] + logging_timeline[i][EP]['intervals'][j])
        # for x in range(1, EP):
        x = i_episode
        logging_timeline[0][x]['UAV_PositionList'] = UAV.PositionList
        logging_timeline[0][x]['UAV_PositionCor'] = UAV.PositionCor
        for i in range(param['num_Devices']):
            logging_timeline[i][x]['intervals'] = Devices[i].intervals
            logging_timeline[i][x]['TimeList'] = Devices[i].TimeList
            logging_timeline[i][x]['KeyRewards'] = Devices[i].KeyReward
            logging_timeline[i][x]['KeyTime'] = Devices[i].KeyTime
            # if not logging_timeline[i][x]['intervals']:
            #     continue
            # logging_timeline[i][x]['timeline'].append(logging_timeline[i][x]['intervals'][0])
            # for j in range(1, len(logging_timeline[i][x]['intervals'])):
            #     logging_timeline[i][x]['timeline'].append(
            #         logging_timeline[i][x]['timeline'][j - 1] + logging_timeline[i][x]['intervals'][j])
            ls1 = [0] + logging_timeline[i][x]['intervals']
            ls2 = logging_timeline[i][x]['KeyRewards']
            logging_timeline[i][x]['avg_reward'] = sum([x * y for x, y in zip(ls1, ls2)]) / logging_timeline[i][x]['KeyTime'][-1]

        # pdb.set_trace()


        print("----------------------------------------------------------------------------")
        print("The percentage to all the devices:")
        for x in range(param['num_Devices']):
            action_list = model.actions[(i_episode - 1) * param['nTimeUnits']::]
            p = len([ele for ele in action_list if ele == x]) / param['nTimeUnits']
            print(f'{x:2}: {100 * p:5.2f}%')
            # print(f'{x:2}: {100 * p:5.2f}%', end = '')
        # print('  ')


        if i_episode % args.log_interval == 0:
            print('Forced: Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, ave_Reward))
        # print("**********************************************************************************")
        # check if we have "solved" the cart pole problem
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break
        if i_episode >= EP:
            break


    for k in range(len(rate1)):
        m1.append(mean(rate1[k]))
        m2.append(mean(rate2[k]))




Ep_reward = []
Ave_Reward = []
m1 = []
m2 = []
EP = param['episodes']

# logging for each episode
# logging_timeline = [ device0, device1, device2....,  ,  ]
# device = [episode0, episode1, episode2, ...,  ]
# episode = {'intervals': [], 'rewards': []}
logging_timeline = []
for i in range(param['num_Devices']):
    logging_timeline.append([])
    for j in range(EP + 1):
        logging_timeline[i].append(
            {'intervals': [], 'rewards': [], 'rewards_regular': [], 'timeline': [], 'plt_reward': [], 'avg_reward': []})
    # logging_timeline.append([{'intervals': [], 'rewards': [], 'rewards_regular': []}, {'intervals': [], 'rewards': [], 'rewards_regular': []}])


def main():

    # log parameters
    print(param)




    # †††††††††††††††††††††††††††††††††††††††Random Trajectory††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    print("Random teajectory: One Episode Only")
    env_random.initialization(Devices_random, UAV_random)
    state_random = env_random.reset(Devices_random, UAV_random)
    ep_reward_random = 0
    t = 0
    n = 0  # logging fly behaviors
    while t < param['nTimeUnits']:
        t = t + 1
        n = n + 1
        action_random = np.random.randint(param['num_Devices'])  # 纯粹随机选择
        state_random, reward_, reward_rest, reward_random = env_random.step(state_random, action_random, t)
        model.rewards_random.append(reward_random)
        ep_reward_random += reward_random
        model.actions_random.append(action_random)
        model.states_random.append(state_random)
        print("Random: The {} episode" " and the {} fly" " at the end of {} time slots. " "Visit device {}".format(1, n,t,action_random))
    ave_Reward_random = ep_reward_random / param['nTimeUnits']
    print('Random: Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(1, ep_reward_random, ave_Reward_random))
    # †††††††††††††††††††††††††††††††††††††††Random Trajectory††††††††††††††††††††††††††††††††††††††††††††††††††††††††††

    learning()

    # # Plotting Phase
    fig, ax = plt.subplots(1)
    # ax[0].plot(np.arange(i_episode), Ep_reward, label='Actor-Critic')
    # ax[0].set_xlabel('Episodes')  # Add an x-label to the axes.
    # ax[0].set_ylabel('ep_reward')  # Add a y-label to the axes.
    # ax[0].set_title("The ep_reward")  # Add a title to the axes.

    # ax.plot(np.arange(1, EP+1), Ave_Reward, label='%.0f  Devices, %.0f TimeUnits, %.0f  episodes' %(param['num_Devices'], param['nTimeUnits'], EP, args.gamma, args.learning_rate))
    ax.plot(np.arange(1, EP+1), Ave_Reward, label= str(param['num_Devices']) + ' Devices,' + str(EP) + ' episodes,' +  str(param['nTimeUnits']) + ' TimeUnits,' +  str(args.gamma) + ' gamma,' + str(args.learning_rate) + ' lr')
    ax.set_xlabel('Episodes')  # Add an x-label to the axes.
    ax.set_ylabel('Ave_Reward')  # Add a y-label to the axes.
    ax.set_title("The Ave_Reward")  # Add a title to the axes.
    ax.axhline(y = max(Ave_Reward), color='r', linestyle='--', linewidth='0.9',   label='Smart: ' + str(max(Ave_Reward)))
    ax.axhline(y = ave_Reward_random, color='b', linestyle='--', linewidth='0.9', label='Random:'+ str(ave_Reward_random))
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
            ax1[i].set_title('CPU Capacity: %.0f' % (Devices[i].cpu_capacity))
            for vv in range(len(np.where(logging_timeline[i][x]['TimeList'])[0])):
                ax1[i].axvline(x=np.where(logging_timeline[i][x]['TimeList'])[0][vv], linestyle='--', linewidth='0.9')
                # ax1[i].plot([np.where(Devices[i].TimeList)],[logging_timeline[i][x]['rewards']], 'o')
    plt.show()
#      https://matplotlib.org/stable/tutorials/text/text_intro.html



if __name__ == '__main__':
    main()