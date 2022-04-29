import argparse
import numpy as np
import random
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt
# import matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from IoTEnv import Uav, Device, Env, Policy
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# Press ⌘F8 to toggle the breakpoint.


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
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
param = {'nTimeUnits': 80, 'num_Devices': 2, 'V': 72, 'field': 1, 'dist': 0.040, 'freq_low': 8, 'freq_high': 16}
Devices = []
# for i in range(param['num_Devices']):
#     Devices.append(Device(random.randint(param['freq_low'], param['freq_high']), random.randint(30, 70), param['field']))
# Devices.append(Device(25, random.randint(30, 70), param['field']))
Devices.append(Device(20, random.randint(30, 70), param['field']))
Devices.append(Device(10, random.randint(30, 70), param['field']))

UAV = Uav(param['V'])
env = Env(Devices, UAV, param['nTimeUnits'])

model = Policy(1 * param['num_Devices']+2, param['num_Devices'])
optimizer = optim.Adam(model.parameters(), lr=3e-2)  # lr=3e-2
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


def main():
    running_reward = 10

    # log parameters
    print(param)
    print('Initial running reward: ', running_reward )

    Ep_reward = []
    Running_reward = []
    Ave_Reward = []

    # run infinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        state = env.reset(Devices, UAV)
        print("the initial state: ", state)
        print("----------------------------------------------------------------------------")
        print("       ")

        model.states.append(state)
        ep_reward = 0
        # FIXME: deepcopy to Devices and UAV for random trajectory
        # Because env.step can change one device and UAV's location

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning

        t = 0
        n = 0  # logging fly behaviors
        # FIXME: when the distance is large or the velocity is small, the Fly_time can be too large to surpass the nTimeUnits
        while t <= param['nTimeUnits']:
        # for t in range(0, param['nTimeUnits']):

            # select action from policy
            action = select_action(state)
            # action = 0

            # take the action
            # state, reward, reward_Regular, t = env.step(state, action, t)
            state, reward_, reward_rest, reward, t = env.step(state, action, t)
            n += 1


            print("the action          ", action)
            print("the state:          ", state)
            print("the reward_         ", reward_)  # current device
            print("the rest reward:    ", reward_rest)  # of other devices
            print("the sum reward:     ", reward)  # reward_ + reward_rest

            model.actions.append(action)
            model.states.append(state)
            model.reward_.append(reward_)
            model.reward_rest.append(reward_rest)
            model.rewards.append(reward)
            ep_reward += reward
            # if done:
            #     break

            print("The {} episode" " and the {} fly" " at the end of {} time slots. " "Visit device {}".format(i_episode, n, t, action))
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

        # log results
        print("**********************************************************************************")
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tRunning reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward, ave_Reward))
        print("**********************************************************************************")
        # check if we have "solved" the cart pole problem
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break
        if i_episode > 9:
            break


    # Plotting Phase
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(np.arange(i_episode), Ep_reward, label='Actor-Critic')
    ax[0].set_xlabel('Episodes')  # Add an x-label to the axes.
    ax[0].set_ylabel('ep_reward')  # Add a y-label to the axes.
    ax[0].set_title("The ep_reward")  # Add a title to the axes.

    ax[1].plot(np.arange(i_episode), Running_reward, label='Actor-Critic')
    ax[1].set_xlabel('Episodes')  # Add an x-label to the axes.
    ax[1].set_ylabel('Running_reward')  # Add a y-label to the axes.
    ax[1].set_title("The Running_reward")  # Add a title to the axes.

    ax[2].plot(np.arange(i_episode), Ave_Reward, label='Actor-Critic')
    ax[2].set_xlabel('Episodes')  # Add an x-label to the axes.
    ax[2].set_ylabel('Ave_Reward')  # Add a y-label to the axes.
    ax[2].set_title("The Ave_Reward")  # Add a title to the axes.
    plt.show()


if __name__ == '__main__':
    main()

