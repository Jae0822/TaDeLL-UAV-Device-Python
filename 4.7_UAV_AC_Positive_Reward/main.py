import argparse
import numpy as np
import random
from itertools import count
from collections import namedtuple
import copy
import matplotlib.pyplot as plt
# import matplotlib
from statistics import mean
from sklearn import preprocessing
import pickle
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# import pdb

from IoTEnv import Uav, Device, Env, Policy
from UAVEnergy import UAV_Energy


with open('SourceTask_temp.pkl', 'rb') as f:
    TaskList_set, Values_array_set = pickle.load(f)



SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'velocity'])


# Prepare the environment and devices
length = 3000
param = {'episodes':25, 'nTimeUnits': length, 'nTimeUnits_random': length, 'nTimeUnits_force': length,
         'gamma': 0.99, 'learning_rate': 0.07, 'log_interval': 1, 'seed': 0, 'alpha': 2, 'mu': 0.5, 'beta': 0.5,
         'num_Devices': 5, 'V': 25, 'V_Lim': 40, 'field': 500, 'dist': 0.040, 'freq_low': 8, 'freq_high': 16}
np.random.seed(param['seed'])
torch.manual_seed(param['seed'])
#torch.device("mps")
torch.set_num_interop_threads(8)
torch.set_num_threads(8)

Devices = []

Devices.append(Device(500, 50, param['field']))
Devices.append(Device(450, 50, param['field']))
Devices.append(Device(400, 50, param['field']))
Devices.append(Device(350, 50, param['field']))
Devices.append(Device(300, 50, param['field']))
# Devices.append(Device(300, 50, param['field']))
# Devices.append(Device(250, 50, param['field']))
# Devices.append(Device(230, 50, param['field']))
# Devices.append(Device(200, 50, param['field']))
# Devices.append(Device(150, 50, param['field']))

# Devices.append(Device(530, 50, param['field']))
# Devices.append(Device(510, 50, param['field']))
# Devices.append(Device(500, 50, param['field']))
# Devices.append(Device(485, 50, param['field']))
# Devices.append(Device(470, 50, param['field']))
# Devices.append(Device(450, 50, param['field']))
# Devices.append(Device(430, 50, param['field']))
# Devices.append(Device(400, 50, param['field']))
# Devices.append(Device(380, 50, param['field']))
# Devices.append(Device(350, 50, param['field']))
# Devices.append(Device(370, 50, param['field']))
# Devices.append(Device(340, 50, param['field']))
# Devices.append(Device(330, 50, param['field']))
# Devices.append(Device(315, 50, param['field']))
# Devices.append(Device(300, 50, param['field']))
# Devices.append(Device(275, 50, param['field']))
# Devices.append(Device(250, 50, param['field']))
# Devices.append(Device(230, 50, param['field']))
# Devices.append(Device(215, 50, param['field']))
# Devices.append(Device(200, 50, param['field']))
# Devices.append(Device(180, 50, param['field']))
# Devices.append(Device(150, 50, param['field']))
# Devices.append(Device(130, 50, param['field']))
# Devices.append(Device(115, 50, param['field']))
# Devices.append(Device(100, 50, param['field']))

UAV = Uav(param['V'], Devices)
env = Env(Devices, UAV, param['nTimeUnits'])
env.initialization(Devices, UAV)

Devices_random = copy.deepcopy(Devices)
UAV_random = copy.deepcopy(UAV)
env_random = Env(Devices_random, UAV_random, param['nTimeUnits_random'])

Devices_force = copy.deepcopy(Devices)
UAV_force = copy.deepcopy(UAV)
env_force = Env(Devices_force, UAV_force, param['nTimeUnits_force'])

model = Policy(param['num_Devices'], param['num_Devices'], param['V_Lim'])
optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])  # lr=3e-2
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    # state = torch.from_numpy(state).float()
    state = torch.from_numpy(state).double()
    probs, state_value, velocity = model(state)

    # create a categorical distribution over the list of probabilities of actions
    print(probs)
    print(state_value)
    print(velocity)
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()
    # action = torch.tensor(2)


    # print(state)
    for label, p in enumerate(probs):
        print(f'{label:2}: {100 * p:5.2f}%')
    # print("---", action, "is chosen")

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value, velocity))

    # the action to take
    return action.item(), velocity.item()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    # velocity_losses = []
    returns = []  # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + param['gamma'] * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value, velocity), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage - velocity * advantage)
        # policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # 尝试添加速度：没必要，因为上面计算policy_losses的时候，就已经有了velocity的部分
        # velocity_losses.append(F.smooth_l1_loss(velocity, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
           # + torch.stack(velocity_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    # del model.reward_[:]
    del model.saved_actions[:]


def learning():

    rate1 = []
    rate2 = []

    # env.initialization(Devices, UAV)

    for i_episode in count(1):

        state = env.reset(Devices, UAV)
        # print("the initial state: ", state)
        print("----------------------------------------------------------------------------")
        print("       ")

        model.states.append(state)
        ep_reward = 0
        t = 0
        n = 0  # logging fly behaviors


        while t < param['nTimeUnits']:

            # select action from policy
            print('state:', state)

            action, velocity = select_action(state)


            # compute the distance
            CPoint = env.UAV.location  # current location
            NPoint = env.Devices[action].location  # next location
            distance = np.linalg.norm(CPoint - NPoint)  # Compute the distance of two points
            Fly_time = 1 if distance == 0 else math.ceil(distance / velocity)
            # t = t + Fly_time
            print(Fly_time)
            PV = UAV_Energy(velocity) * Fly_time


            # take the action
            # state, reward, reward_Regular, t = env.step(state, action, t)
            t = t + Fly_time
            state, reward_, reward_rest, reward = env.step(state, action, velocity, t, PV, param, Fly_time)
            print(reward_)
            print(reward_rest)
            print(reward)
            n += 1







            # print("the action          ", action)
            # print("the state:          ", state)
            # print("the reward_         ", reward_)  # current device
            # print("the rest reward:    ", reward_rest)  # of other devices
            # print("the sum reward:     ", reward)  # reward_ + reward_rest

            model.actions.append(action)
            model.states.append(state)


            # model.reward_.append(reward_)
            # model.reward_rest.append(reward_rest)
            # model.reward_.append(reward)
            model.rewards.append(reward)
            ep_reward += reward

            print("Smart: The {} episode" " and the {} fly" " at the end of {} time slots. " "Visit device {}".format(i_episode, n, t, action))

            print("----------------------------------------------------------------------------")
            print("       ")


        # 归一化
        # model.rewards = preprocessing.normalize([model.reward_])[0]
        # model.rewards = model.rewards.tolist()
        # model.rewards = [np.float64(x) for x in model.rewards]

        #  Average Reward
        # FIXME: 为什么要除以N呀？UAV飞行的次数每次都不一样咋办？直接不除以N不就好了？或者除以设备数量？
        """
        FX3
        不除以N，凸起变高
        """

        ave_Reward = ep_reward
        # ave_Reward = sum(model.rewards) / n

        # update cumulative reward
        # running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        Ep_reward.append(ep_reward / n)
        # Running_reward.append(running_reward)
        Ave_Reward.append(ep_reward)

        x = i_episode
        logging_timeline[0][x]['UAV_TimeList'] = UAV.TimeList
        logging_timeline[0][x]['UAV_PositionList'] = UAV.PositionList
        logging_timeline[0][x]['UAV_PositionCor'] = UAV.PositionCor
        logging_timeline[0][x]['UAV_VelocityList'] = UAV.VelocityList
        logging_timeline[0][x]['UAV_Reward'] = UAV.Reward  # 设备的COST(AOI+CPU)，负数，绝对值越小越好
        logging_timeline[0][x]['UAV_Energy'] = UAV.Energy  # UAV的飞行能量，正数，绝对值越小越好
        logging_timeline[0][x]['UAV_R_E'] = UAV.Sum_R_E  # 上面两项（REWARD+ENERGY）的和，负数，绝对值越小越好（这个是STEP输出的最后一个REWARD，优化目标本标，优化的是每个EPISODE的均值：Ep_reward）
        logging_timeline[0][x]['UAV_AoI'] = UAV.AoI  # 设备的AOI，正数，绝对值越小越好
        logging_timeline[0][x]['UAV_CPU'] = UAV.CPU  # 设备的CPU，正数，绝对值越小越好
        logging_timeline[0][x]['UAV_b'] = UAV.b      # 设备的B，正数，绝对值越小越好
        for i in range(param['num_Devices']):
            logging_timeline[i][x]['intervals'] = Devices[i].intervals
            logging_timeline[i][x]['TimeList'] = Devices[i].TimeList
            logging_timeline[i][x]['TaskList'] = Devices[i].TaskList
            logging_timeline[i][x]['KeyTime'] = Devices[i].KeyTime
            # 记录每一个EPISODE的非REGULAR的数据
            # FIXME: 这里的KEYREWARD只包含了AOI+CPU，没有包含UAV的能耗PV
            # 这里每一个DEVICE只能这样啊，DEVICE不像UAV一样一下一下飞，DEVICE是每一个时隙的
            # 这里的KEYREWARD是上面step输出reward的一部分，不包括UAV的PV，减200的penalty，不包括reward_rest
            logging_timeline[i][x]['KeyTsk'] = Devices[i].KeyTsk
            logging_timeline[i][x]['KeyPol'] = Devices[i].KeyPol
            logging_timeline[i][x]['KeyRewards'] = Devices[i].KeyReward
            logging_timeline[i][x]['KeyAoI'] = Devices[i].KeyAoI
            logging_timeline[i][x]['KeyCPU'] = Devices[i].KeyCPU
            logging_timeline[i][x]['Keyb'] = Devices[i].Keyb
            # 记录对应的REGULAR的数据
            logging_timeline[i][x]['KeyTsk_Regular'] = Devices[i].KeyTsk_Regular
            logging_timeline[i][x]['KeyPol_Regular'] = Devices[i].KeyPol_Regular
            logging_timeline[i][x]['KeyReward_Regular'] = Devices[i].KeyReward_Regular
            logging_timeline[i][x]['KeyAoI_Regular'] = Devices[i].KeyAoI_Regular
            logging_timeline[i][x]['KeyCPU_Regular'] = Devices[i].KeyCPU_Regular
            logging_timeline[i][x]['Keyb_Regular'] = Devices[i].Keyb_Regular

            # if not logging_timeline[i][x]['intervals']:
            #     continue
            # logging_timeline[i][x]['timeline'].append(logging_timeline[i][x]['intervals'][0])
            # for j in range(1, len(logging_timeline[i][x]['intervals'])):
            #     logging_timeline[i][x]['timeline'].append(
            #         logging_timeline[i][x]['timeline'][j - 1] + logging_timeline[i][x]['intervals'][j])
            ls1 = [0] + logging_timeline[i][x]['intervals']
            ls2 = logging_timeline[i][x]['KeyRewards']
            # 这里的avg_reward知识单纯的每一个device的reward均值
            if len(logging_timeline[i][x]['KeyTime']) == 1:
                logging_timeline[i][x]['avg_reward'] = None
            else:
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


        if i_episode % param['log_interval'] == 0:
            print('Smart: Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
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


def painting(avg):
    fig0, ax0 = plt.subplots(1)
    [plt.scatter(D.location[0], D.location[1]) for D in env.Devices]
    x = [D.location[0] for D in env.Devices]
    y = [D.location[1] for D in env.Devices]
    No = list(range(len(env.Devices)))
    # ax.scatter(x, y)
    for i, txt in enumerate(No):
        ax0.annotate(txt, (x[i], y[i]))
    ax0.plot([0],[0], label = 'V_Lim:' + str(param['V_Lim']) + ',  V:' + str(param['V']))
    ax0.set_xlim(0, 1000)
    ax0.set_ylim(0, 1000)
    ax0.legend(loc="best")
    ax0.grid(True)


    # †††††††††††††††††††††††††††††††††††††††Plotting Phase††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    fig, ax = plt.subplots(1)
    # ax[0].plot(np.arange(i_episode), Ep_reward, label='Actor-Critic')
    # ax[0].set_xlabel('Episodes')  # Add an x-label to the axes.
    # ax[0].set_ylabel('ep_reward')  # Add a y-label to the axes.
    # ax[0].set_title("The ep_reward")  # Add a title to the axes.

    # ax.plot(np.arange(1, EP+1), Ave_Reward, label='%.0f  Devices, %.0f TimeUnits, %.0f  episodes' %(param['num_Devices'], param['nTimeUnits'], EP, args.gamma, args.learning_rate))
    ax.plot(np.arange(1, param['episodes'] + 1), avg['Ave_Reward'],
            label=str(param['num_Devices']) + ' Devices,' + str(param['episodes']) + ' episodes,' + str(
                param['nTimeUnits']) + ' TimeUnits,' + str(param['gamma']) + ' gamma,' + str(
                param['learning_rate']) + ' lr,' + str(param['alpha']) + ' alpha, ' + str(param['mu']) + ' mu')
    ax.set_xlabel('Episodes')  # Add an x-label to the axes.
    ax.set_ylabel('Ave_Reward')  # Add a y-label to the axes.
    ax.set_title("The Ave_Reward, NN:" + str(model.pattern))  # Add a title to the axes.
    ax.axhline(y=max(avg['Ave_Reward']), color='r', linestyle='--', linewidth='0.9', label='Smart: ' + str(max(avg['Ave_Reward'])))
    ax.axhline(y=avg['ave_Reward_random'] * len(env_random.UAV.Reward), color='b', linestyle='--', linewidth='0.9',
               label='Random:' + str(avg['ave_Reward_random']*len(env_random.UAV.Reward)))
    ax.axhline(y=avg['ave_Reward_force'] * len(env_force.UAV.Reward), color='g', linestyle='--', linewidth='0.9', label='Forced:' + str(avg['ave_Reward_force']* len(env_force.UAV.Reward)))
    ax.legend(loc="best")


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


    # †††††††††††††††††††††††††††††††††††††††Smart††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    x = param['episodes']
    fig1, ax1 = plt.subplots(param['num_Devices'])
    fig1.supxlabel('Time Unites for one episode')
    fig1.supylabel('The Ave Reward')
    fig1.suptitle('The episode %.0f' % (x))
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
            # ax1[i].set_title('CPU Capacity: %.0f' % (env.Devices[i].cpu_capacity))
            for vv in range(len(np.where(logging_timeline[i][x]['TimeList'])[0])):
                ax1[i].axvline(x=np.where(logging_timeline[i][x]['TimeList'])[0][vv], linestyle='--', linewidth='0.9')
                # ax1[i].plot([np.where(Devices[i].TimeList)],[logging_timeline[i][x]['rewards']], 'o')
    # plt.show()
    #      https://matplotlib.org/stable/tutorials/text/text_intro.html

    # †††††††††††††††††††††††††††††††††††††††RANDOM††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    fig3, ax3 = plt.subplots(param['num_Devices'])
    fig3.supxlabel('Time Unites for one episode')
    fig3.supylabel('The Ave Reward')
    fig3.suptitle('The Random')
    for i in range(param['num_Devices']):
        if len(logging_timeline[i][0]['Random_intervals']) == 0:
            ax3[i].text(0.4, 0.5, 'No Visit by UAV')
        else:
            ax3[i].step(logging_timeline[i][0]['Random_KeyTime'], logging_timeline[i][0]['Random_KeyRewards'], '^-g',
                        where='post',
                        label='device %.0f' % (i))
            ax3[i].set_xlim([0, param['nTimeUnits']])
            ax3[i].axhline(y=logging_timeline[i][0]['Random_avg_reward'], color='r', linestyle='--', linewidth='0.9',
                           label=logging_timeline[i][0]['Random_avg_reward'])
            ax3[i].legend(loc="best")
            # ax1[i].text(2, logging_timeline[i][x]['avg_reward'], logging_timeline[i][x]['avg_reward'], verticalalignment='bottom',horizontalalignment='left', rotation=360, color='r')
            ax3[i].set_ylabel('device  %.0f' % (i))
            # if i == 0:
            #     ax1[i].set_title(model.pattern)
            ax3[i].set_title('CPU Capacity: %.0f' % (env_random.Devices[i].cpu_capacity))
            for vv in range(len(np.where(logging_timeline[i][0]['Random_TimeList'])[0])):
                ax3[i].axvline(x=np.where(logging_timeline[i][0]['Random_TimeList'])[0][vv], linestyle='--',
                               linewidth='0.9')
                # ax1[i].plot([np.where(Devices[i].TimeList)],[logging_timeline[i][x]['rewards']], 'o')


    # †††††††††††††††††††††††††††††††††††††††Force††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    fig4, ax4 = plt.subplots(param['num_Devices'])
    fig4.supxlabel('Time Unites for one episode')
    fig4.supylabel('The Ave Reward')
    fig4.suptitle('The Force')
    for i in range(param['num_Devices']):
        if len(logging_timeline[i][0]['Force_intervals']) == 0:
            ax4[i].text(0.4, 0.5, 'No Visit by UAV')
        else:
            ax4[i].step(logging_timeline[i][0]['Force_KeyTime'], logging_timeline[i][0]['Force_KeyRewards'], '^-g',
                        where='post',
                        label='device %.0f' % (i))
            ax4[i].set_xlim([0, param['nTimeUnits']])
            ax4[i].axhline(y=logging_timeline[i][0]['Force_avg_reward'], color='r', linestyle='--', linewidth='0.9',
                           label=logging_timeline[i][0]['Force_avg_reward'])
            ax4[i].legend(loc="best")
            # ax1[i].text(2, logging_timeline[i][x]['avg_reward'], logging_timeline[i][x]['avg_reward'], verticalalignment='bottom',horizontalalignment='left', rotation=360, color='r')
            ax4[i].set_ylabel('device  %.0f' % (i))
            # if i == 0:
            #     ax1[i].set_title(model.pattern)
            ax4[i].set_title('CPU Capacity: %.0f' % (env_force.Devices[i].cpu_capacity))
            for vv in range(len(np.where(logging_timeline[i][0]['Force_TimeList'])[0])):
                # ax1[i].plot([np.where(Devices[i].TimeList)],[logging_timeline[i][x]['rewards']], 'o')
                ax4[i].axvline(x=np.where(logging_timeline[i][0]['Force_TimeList'])[0][vv], linestyle='--',
                               linewidth='0.9')

    d = 1
    # †††††††††††††††††††††††††††††††††††††††柱状图††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    x = param['episodes']
    fig7, ax7 = plt.subplots(2, sharex=True)
    fig7.suptitle('Devcie and UAV cost')
    type = ['Random', 'Force', 'Smart']
    data111 = [-np.mean(env_random.UAV.Reward), -np.mean(env_force.UAV.Reward),
               -np.mean(logging_timeline[0][x]['UAV_Reward'])]
    data1111 = [-np.sum(env_random.UAV.Reward), -np.sum(env_force.UAV.Reward),
                -np.sum(logging_timeline[0][x]['UAV_Reward'])]
    data22 = [np.mean(env_random.UAV.Energy), np.mean(env_force.UAV.Energy),
              np.mean(logging_timeline[0][x]['UAV_Energy'])]
    data222 = [np.sum(env_random.UAV.Energy), np.sum(env_force.UAV.Energy),
               np.sum(logging_timeline[0][x]['UAV_Energy'])]
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


    x = param['episodes']
    fig8, ax8 = plt.subplots(2, sharex=True)
    fig8.suptitle('AoI and CPU cost')
    type = ['Random', 'Force', 'Smart']
    dataAoImean = [np.mean(env_random.UAV.AoI), np.mean(env_force.UAV.AoI),
               np.mean(logging_timeline[0][x]['UAV_AoI'])]
    dataAoIsum = [np.sum(env_random.UAV.AoI), np.sum(env_force.UAV.AoI),
                np.sum(logging_timeline[0][x]['UAV_AoI'])]
    dataCPUmean = [np.mean(env_random.UAV.CPU), np.mean(env_force.UAV.CPU),
              np.mean(logging_timeline[0][x]['UAV_CPU'])]
    dataCPUsum = [np.sum(env_random.UAV.CPU), np.sum(env_force.UAV.CPU),
               np.sum(logging_timeline[0][x]['UAV_CPU'])]
    ax8[0].bar(type, [k * param['beta'] for k in dataAoImean], label='AoI')
    ax8[0].bar(type, [k * param['beta'] for k in dataCPUmean], bottom=np.array(dataAoImean) * param['beta'], label='CPU')
    ax8[0].axhline(y=0, color='k', linestyle='-', linewidth='0.6')
    ax8[0].legend(loc="best")
    ax8[0].set_title('The Mean')
    ax8[0].set_ylabel('Total Cost')  # Add a y-label to the axes.
    ax8[1].bar(type, [k * param['beta'] for k in dataAoIsum], label='AoI')
    ax8[1].bar(type, [k * param['beta'] for k in dataCPUsum], bottom=np.array(dataAoIsum) * param['beta'], label='CPU')
    ax8[1].axhline(y=0, color='k', linestyle='-', linewidth='0.6')
    ax8[1].legend(loc="best")
    ax8[1].set_title('The Sum')
    ax8[1].set_ylabel('Total Cost')  # Add a y-label to the axes.
    # plt.show()

    # †††††††††††††††††††††††††††††††††††††††速度图††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    V_avg = []
    for x in range(1, param['episodes']):
        V_avg.append(mean(logging_timeline[0][x]['UAV_VelocityList']))
    fig_v, ax_v = plt.subplots(1)
    ax_v.plot(np.arange(1, param['episodes']), V_avg)
    ax_v.set_ylabel('Velocity(m/s)')
    ax_v.set_xlabel('episodes')
    ax_v.grid(True)
    fig_v.suptitle('Velocity trend')
    plt.show()


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

    # †††††††††††††††††††††††††††††††††††††††Smart Trajectory††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    learning()
    # †††††††††††††††††††††††††††††††††††††††Smart Trajectory††††††††††††††††††††††††††††††††††††††††††††††††††††††††††



    # †††††††††††††††††††††††††††††††††††††††Random Trajectory††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    print("Random trajectory: One Episode Only")
    # env_random.initialization(Devices_random, UAV_random)
    state_random = env_random.reset(Devices_random, UAV_random)
    ep_reward_random = 0
    t = 0
    n = 0  # logging fly behaviors
    Reward_random = []
    PV_random = []
    while t < param['nTimeUnits_random']:
        action_random = np.random.randint(param['num_Devices'])  # 纯粹随机选择
        CPoint = env_random.UAV.location  # current location
        NPoint = env_random.Devices[action_random].location  # next location
        distance = np.linalg.norm(CPoint - NPoint)  # Compute the distance of two points
        # env_random.UAV.V = logging_timeline[0][param['episodes']]['UAV_VelocityList'][-1]
        Fly_time = 1 if distance == 0 else math.ceil(distance / env_random.UAV.V)
        PV = UAV_Energy(param['V']) * Fly_time
        t = t + Fly_time
        if t > param['nTimeUnits_random']:
            break
        n = n + 1
        state_random, reward_, reward_rest, reward_random  = env_random.step(state_random, action_random, param['V'], t, PV, param, Fly_time)
        # model.rewards_random.append(reward_random)
        PV_random.append(PV)
        Reward_random.append(reward_random)
        ep_reward_random += reward_random
        # model.actions_random.append(action_random)
        # model.states_random.append(state_random)
        print("Random: The {} episode" " and the {} fly" " at the end of {} time slots. " "Visit device {}".format(1, n,
                                                                                                                   t,
                                                                                                                   action_random))

    logging_timeline[0][0]['Reward_random'] = Reward_random
    logging_timeline[0][0]['Random_UAV_TimeList'] = UAV_random.TimeList
    logging_timeline[0][0]['Random_UAV_PositionList'] = UAV_random.PositionList
    logging_timeline[0][0]['Random_UAV_PositionCor'] = UAV_random.PositionCor
    logging_timeline[0][0]['Random_UAV_VelocityList'] = UAV_random.VelocityList
    logging_timeline[0][0]['Random_UAV_Reward'] = UAV_random.Reward
    logging_timeline[0][0]['Random_UAV_Energy'] = UAV_random.Energy
    logging_timeline[0][0]['Random_UAV_R_E'] = UAV_random.Sum_R_E
    logging_timeline[0][0]['Random_UAV_AoI'] = UAV_random.AoI
    logging_timeline[0][0]['Random_UAV_CPU'] = UAV_random.CPU
    logging_timeline[0][0]['Random_UAV_b'] = UAV_random.b
    for i in range(param['num_Devices']):
        logging_timeline[i][0]['Random_intervals'] = Devices_random[i].intervals
        logging_timeline[i][0]['Random_TimeList'] = Devices_random[i].TimeList
        logging_timeline[i][0]['Random_KeyTime'] = Devices_random[i].KeyTime
        logging_timeline[i][0]['Random_TaskList'] = Devices_random[i].TaskList
        # 记录每一个EPISODE的非REGULAR的数据
        logging_timeline[i][0]['Random_KeyTsk'] = Devices_random[i].KeyTsk
        logging_timeline[i][0]['Random_KeyPol'] = Devices_random[i].KeyPol
        logging_timeline[i][0]['Random_KeyRewards'] = Devices_random[i].KeyReward
        logging_timeline[i][0]['Random_KeyAoI'] = Devices_random[i].KeyAoI
        logging_timeline[i][0]['Random_KeyCPU'] = Devices_random[i].KeyCPU
        logging_timeline[i][0]['Random_Keyb'] = Devices_random[i].Keyb
        # 记录对应的REGULAR的数据
        logging_timeline[i][0]['Random_KeyTsk_Regular'] = Devices_random[i].KeyTsk_Regular
        logging_timeline[i][0]['Random_KeyPol_Regular'] = Devices_random[i].KeyPol_Regular
        logging_timeline[i][0]['Random_KeyReward_Regular'] = Devices_random[i].KeyReward_Regular
        logging_timeline[i][0]['Random_KeyAoI_Regular'] = Devices_random[i].KeyAoI_Regular
        logging_timeline[i][0]['Random_KeyCPU_Regular'] = Devices_random[i].KeyCPU_Regular
        logging_timeline[i][0]['Random_Keyb_Regular'] = Devices_random[i].Keyb_Regular
        ls1 = [0] + logging_timeline[i][0]['Random_intervals']
        ls2 = logging_timeline[i][0]['Random_KeyRewards']
        if len(logging_timeline[i][0]['Random_KeyTime']) == 1:
            logging_timeline[i][0]['Random_avg_reward'] = None
        else:
            logging_timeline[i][0]['Random_avg_reward'] = sum([x * y for x, y in zip(ls1, ls2)]) / \
                                                   logging_timeline[i][0]['Random_KeyTime'][-1]
    ave_Reward_random = ep_reward_random / n
    print('Random: Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(1, ep_reward_random, ave_Reward_random))
    # †††††††††††††††††††††††††††††††††††††††Random Trajectory††††††††††††††††††††††††††††††††††††††††††††††††††††††††††



    # †††††††††††††††††††††††††††††††††††††††Forced Trajectory††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    print("Forced trajectory: One Episode Only")
    # env_force.initialization(Devices_force, UAV_force)
    state_force = env_force.reset(Devices_force, UAV_force)
    ep_reward_force = 0
    t = 0
    n = 0
    Reward_force = []
    PV_force = []
    while t < param['nTimeUnits_force']:

        # 强制选择action
        action_table_force = np.zeros(param['num_Devices'])  # 筛选出当前有新任务的设备
        for i in range(param['num_Devices']):
            if Devices_force[i].TimeList[t-1] == 1:
                action_table_force[i] = 1
        inx = np.where(action_table_force == 1)[0]
        # action = inx[np.random.randint(len(inx))] if inx else np.random.randint(param['num_Devices']) # 随机选一个去访问
        if inx.any():
            action_force = inx[0]  # inx[np.random.randint(len(inx))] #优先选择变化最不频繁的
        else:
            action_force = np.random.randint(param['num_Devices'])
        # compute the distance
        CPoint = env_force.UAV.location  # current location
        NPoint = env_force.Devices[action_force].location  # next location
        distance = np.linalg.norm(CPoint - NPoint)  # Compute the distance of two points
        # env_force.UAV.V = logging_timeline[0][param['episodes']]['UAV_VelocityList'][-1]
        Fly_time = 1 if distance == 0 else math.ceil(distance / env_force.UAV.V)
        PV = UAV_Energy(param['V']) * Fly_time
        t = t + Fly_time
        if t > param['nTimeUnits_force']:
            break
        n = n + 1
        state_force, reward_, reward_rest, reward_force = env_force.step(state_force, action_force, param['V'], t, PV, param, Fly_time)
        PV_force.append(PV)
        Reward_force.append(reward_force)
        ep_reward_force += reward_force
        print("Force: The {} episode" " and the {} fly" " at the end of {} time slots. " "Visit device {}".format(1, n,t,action_force))
    logging_timeline[0][0]['Reward_force'] = Reward_force
    logging_timeline[0][0]['Force_UAV_TimeList'] = UAV_force.TimeList
    logging_timeline[0][0]['Force_UAV_PositionList'] = UAV_force.PositionList
    logging_timeline[0][0]['Force_UAV_PositionCor'] = UAV_force.PositionCor
    logging_timeline[0][0]['Force_UAV_VelocityList'] = UAV_force.VelocityList
    logging_timeline[0][0]['Force_UAV_Reward'] = UAV_force.Reward
    logging_timeline[0][0]['Force_UAV_Energy'] = UAV_force.Energy
    logging_timeline[0][0]['Force_UAV_R_E'] = UAV_force.Sum_R_E
    logging_timeline[0][0]['Force_UAV_AoI'] = UAV_force.AoI
    logging_timeline[0][0]['Force_UAV_CPU'] = UAV_force.CPU
    logging_timeline[0][0]['Force_UAV_b'] = UAV_force.b
    for i in range(param['num_Devices']):
        logging_timeline[i][0]['Force_intervals'] = Devices_force[i].intervals
        logging_timeline[i][0]['Force_TimeList'] = Devices_force[i].TimeList
        logging_timeline[i][0]['Force_KeyTime'] = Devices_force[i].KeyTime
        logging_timeline[i][0]['Force_TaskList'] = Devices_force[i].TaskList
        # 记录每一个EPISODE的非REGULAR的数据
        logging_timeline[i][0]['Force_KeyTsk'] = Devices_force[i].KeyTsk
        logging_timeline[i][0]['Force_KeyPol'] = Devices_force[i].KeyPol
        logging_timeline[i][0]['Force_KeyRewards'] = Devices_force[i].KeyReward
        logging_timeline[i][0]['Force_KeyAoI'] = Devices_force[i].KeyAoI
        logging_timeline[i][0]['Force_KeyCPU'] = Devices_force[i].KeyCPU
        logging_timeline[i][0]['Force_Keyb'] = Devices_force[i].Keyb
        # 记录对应的REGULAR的数据
        logging_timeline[i][0]['Force_KeyTsk_Regular'] = Devices_force[i].KeyTsk_Regular
        logging_timeline[i][0]['Force_KeyPol_Regular'] = Devices_force[i].KeyPol_Regular
        logging_timeline[i][0]['Force_KeyReward_Regular'] = Devices_force[i].KeyReward_Regular
        logging_timeline[i][0]['Force_KeyAoI_Regular'] = Devices_force[i].KeyAoI_Regular
        logging_timeline[i][0]['Force_KeyCPU_Regular'] = Devices_force[i].KeyCPU_Regular
        logging_timeline[i][0]['Force_Keyb_Regular'] = Devices_force[i].Keyb_Regular
        ls1 = [0] + logging_timeline[i][0]['Force_intervals']
        ls2 = logging_timeline[i][0]['Force_KeyRewards']
        if len(logging_timeline[i][0]['Force_KeyTime']) == 1:
            logging_timeline[i][0]['Force_avg_reward'] = None
        else:
            logging_timeline[i][0]['Force_avg_reward'] = sum([x * y for x, y in zip(ls1, ls2)]) / \
                                                   logging_timeline[i][0]['Force_KeyTime'][-1]
    ave_Reward_force = ep_reward_force / n
    print('Force: Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(1, ep_reward_force, ave_Reward_force))
    # †††††††††††††††††††††††††††††††††††††††Forced Trajectory††††††††††††††††††††††††††††††††††††††††††††††††††††††††††






    avg = {}
    avg['Ave_Reward'] = Ave_Reward
    avg['Ep_reward'] = Ep_reward
    avg['ave_Reward_random'] = ave_Reward_random
    avg['ave_Reward_force'] = ave_Reward_force
    # with open('fig_temp.pkl', 'wb') as f:
    #     pickle.dump([model, env, param, avg, logging_timeline], f)

    with open('fig_temp.pkl', 'wb') as f:
        pickle.dump([model, env, env_random, env_force, param, avg, logging_timeline], f)
    # with open('fig_temp.pkl', 'rb') as f:
    #     model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)

    # †††††††††††††††††††††††††††††††††††††††Painting††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    painting(avg)
    # †††††††††††††††††††††††††††††††††††††††Painting††††††††††††††††††††††††††††††††††††††††††††††††††††††††††


    dff = [1]
    luck = [1,2,]



if __name__ == '__main__':
    main()