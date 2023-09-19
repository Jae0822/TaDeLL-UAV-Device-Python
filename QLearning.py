import math
import torch
import torch.nn as nn
import numpy as np
import copy
import random
from collections import namedtuple, deque

from IoTEnv import Uav, Env, QLPolicy
from UAVEnergy import UAV_Energy
import Util

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size = 64):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    

class QLearning:
    def __init__(self, param, logging_timeline) -> None:
        self.param = param
        self.logging_timeline = logging_timeline
        self.devices = Util.initialize_fixed_devices(param, 'tadell')
        self.velocity = param["V"]

        self.uav = Uav(param['V'], self.devices)
        self.env = Env(self.devices, self.uav, param['nTimeUnits'])
        self.env_pgrl = Env(Util.initialize_fixed_devices(param, 'pg_rl'), copy.deepcopy(self.uav), param['nTimeUnits'], 'pg_rl')
        self.actor_model = QLPolicy(param['num_Devices'], param['num_Devices'])
        self.optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=param['learning_rate'])  # lr=3e-2
        # for plotting
        self.Ep_reward = []
        self.Ave_reward = []
        self.Ep_reward_pgrl = []
        self.Ave_reward_pgrl = []
        self.memory = ReplayBuffer(100000)
        self.EPS_END = 0.05
        self.EPS_START = 0.9
        self.EPS_DECAY = 1000
        self.EPS_NUMBER = 0

    def select_action(self, state):
        state = torch.from_numpy(state).double()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.EPS_NUMBER/self.EPS_DECAY)
        self.EPS_NUMBER += 1
        # state = torch.from_numpy(state).float()
        if np.random.rand() < eps_threshold:
            action = random.randrange(len(self.devices))
            velocity = self.velocity
            return action, velocity

        with torch.no_grad():
            probs = self.actor_model(state)
            velocity = self.velocity
            action = torch.argmax(probs)

            # create a categorical distribution over the list of probabilities of actions
            print("select action")
            print("\tprobs: {}".format(probs))
            print("\tstate: {}".format(state))
            print("\tvelocity: {}".format(velocity))

            # save to action buffer
            savedAction = namedtuple('SavedAction', ['log_prob', 'value', 'velocity'])
            self.actor_model.saved_actions.append(savedAction(
                action, state, velocity))

            # the action to take
            return action.item(), velocity

    def finish_episode(self):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """

        batch = self.memory.sample()
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(states, dtype=torch.double)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.double)
        next_states = torch.tensor(next_states, dtype=torch.double)

        q_values = self.actor_model(states)
        q_values = torch.gather(q_values, dim=1, index=actions.unsqueeze(1))
        q_values_next = self.actor_model(next_states)

        target_q_values = rewards + 0.99*torch.max(q_values_next, dim=1).values

        loss = nn.MSELoss()(q_values.squeeze(), target_q_values.detach())
        print("\tq_values: {} target: {} loss: {}".format(q_values, target_q_values, loss))

        # reset gradients
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del self.actor_model.rewards[:]
        del self.actor_model.saved_actions[:]


    def learning(self):

        for i_episode in range(1, self.param['episodes'] + 1):

            self.memory = ReplayBuffer(100000)
            state = self.env.reset()
            self.env_pgrl.reset()
            # print("the initial state: ", state)
            print("----------------------------------------------------------------------------")
            print("       ")

            ep_reward = 0
            ep_reward_pgrl = 0
            t = 0
            self.n_fly = 0  # logging fly behaviors

            while t < self.param['nTimeUnits']:
            # for t in range(0, param['nTimeUnits']):

                # select action from policy
                print("----------------------------------------------------------------------------")
                print('state:', state)

                action, velocity = self.select_action(state)
                velocity = self.velocity
                CPoint = self.env.UAV.location  # current location
                NPoint = self.env.Devices[action].location  # next location
                distance = np.linalg.norm(CPoint - NPoint)  # Compute the distance of two points
                Fly_time = 1 if distance == 0 else math.ceil(distance / velocity)
                # t = t + Fly_time
                PV = UAV_Energy(velocity) * Fly_time
                print("action: {}, cur_loc: {}, next_loc: {}, distance: {}, velocity: {}, fly_time: {}, PV: {}".format(
                    action, CPoint, NPoint, distance, velocity, Fly_time, PV))


                # take the action
                # state, reward, reward_Regular, t = env.step(state, action, t)
                t = t + Fly_time
                next_state, reward_, reward_rest, reward = self.env.step(state, action, velocity, t, PV, self.param, Fly_time)
                print("reward: {}, reward_rest: {}, reward_: {}".format(reward, reward_rest, reward_))
                self.n_fly += 1
                self.memory.add(Experience(state, action, reward, next_state))

                ep_reward += reward

                print("Smart: The {} episode" " and the {} fly" " at the end of {} time slots. " "Visit device {}".format(i_episode, self.n_fly, t, action))

                print("----------------------------------------------------------------------------")
                print("       ")

                state = next_state
                _, _, _, reward = self.env_pgrl.step(state, action, velocity, t, PV, self.param, Fly_time)
                ep_reward_pgrl += reward



            """
            FX3
            不除以N，凸起变高
            """

            self.Ave_reward.append(ep_reward/self.n_fly)
            self.Ep_reward.append(ep_reward)
            self.Ave_reward_pgrl.append(ep_reward_pgrl/self.n_fly)
            self.Ep_reward_pgrl.append(ep_reward_pgrl)

            # perform backprop
            self.finish_episode()

            # save results in logging
            self.save_logging(i_episode)

            if i_episode % self.param['log_interval'] == 0:
                print('Smart: Episode {}\tLast reward: {:.2f}'.format(
                      i_episode, ep_reward))
        
    def save_logging(self, episode):
        self.logging_timeline[0][episode]['UAV_TimeList'] = self.uav.TimeList
        self.logging_timeline[0][episode]['UAV_PositionList'] = self.uav.PositionList
        self.logging_timeline[0][episode]['UAV_PositionCor'] = self.uav.PositionCor
        self.logging_timeline[0][episode]['UAV_VelocityList'] = self.uav.VelocityList
        self.logging_timeline[0][episode]['UAV_Reward'] = self.uav.Reward  # 设备的COST(AOI+CPU)，负数，绝对值越小越好
        self.logging_timeline[0][episode]['UAV_Reward_Regular'] = self.env_pgrl.UAV.Reward  # 设备的COST(AOI+CPU)，负数，绝对值越小越好
        self.logging_timeline[0][episode]['UAV_Energy'] = self.uav.Energy  # UAV的飞行能量，正数，绝对值越小越好
        self.logging_timeline[0][episode]['UAV_R_E'] = self.uav.Sum_R_E  # 上面两项（REWARD+ENERGY）的和，负数，绝对值越小越好（这个是STEP输出的最后一个REWARD，优化目标本标，优化的是每个EPISODE的均值：Ep_reward）
        self.logging_timeline[0][episode]['UAV_AoI'] = self.uav.AoI  # 设备的AOI，正数，绝对值越小越好
        self.logging_timeline[0][episode]['UAV_CPU'] = self.uav.CPU  # 设备的CPU，正数，绝对值越小越好
        self.logging_timeline[0][episode]['UAV_b'] = self.uav.b      # 设备的B，正数，绝对值越小越好
        for i in range(self.param['num_Devices']):
            self.logging_timeline[i][episode]['TaskList'] = self.devices[i].TaskList
            self.logging_timeline[i][episode]['KeyTime'] = self.devices[i].KeyTime
            self.logging_timeline[i][episode]['KeyAoI'] = self.devices[i].KeyAoI
            self.logging_timeline[i][episode]['KeyCPU'] = self.devices[i].KeyCPU
            self.logging_timeline[i][episode]['Keyb'] = self.devices[i].Keyb
            # 记录每一个EPISODE的非REGULAR的数据
            # FIXME: 这里的KEYREWARD只包含了AOI+CPU，没有包含UAV的能耗PV
            # 这里每一个DEVICE只能这样啊，DEVICE不像UAV一样一下一下飞，DEVICE是每一个时隙的
            # 这里的KEYREWARD是上面step输出reward的一部分，不包括UAV的PV，减200的penalty，不包括reward_rest
            self.logging_timeline[i][episode]['KeyRewards'] = self.devices[i].KeyReward
            ls2 = self.logging_timeline[i][episode]['KeyRewards']
            # 这里的avg_reward知识单纯的每一个device的reward均值
            if len(self.logging_timeline[i][episode]['KeyTime']) == 1:
                self.logging_timeline[i][episode]['avg_reward'] = None
            else:
                self.logging_timeline[i][episode]['avg_reward'] = sum(ls2)/len(ls2)
            #####################  REGULAR ########################################
            self.logging_timeline[i][episode]['TaskList_Regular'] = self.env_pgrl.Devices[i].TaskList
            self.logging_timeline[i][episode]['KeyTime_Regular'] = self.env_pgrl.Devices[i].KeyTime
            self.logging_timeline[i][episode]['KeyAoI_Regular'] = self.env_pgrl.Devices[i].KeyAoI
            self.logging_timeline[i][episode]['KeyCPU_Regular'] = self.env_pgrl.Devices[i].KeyCPU
            self.logging_timeline[i][episode]['Keyb_Regular'] = self.env_pgrl.Devices[i].Keyb
            self.logging_timeline[i][episode]['KeyReward_Regular'] = self.env_pgrl.Devices[i].KeyReward
            ls2 = self.logging_timeline[i][episode]['KeyReward_Regular']
            if len(self.logging_timeline[i][episode]['KeyTime_Regular']) == 1:
                self.logging_timeline[i][episode]['avg_reward_Regular'] = None
            else:
                self.logging_timeline[i][episode]['avg_reward_Regular'] = sum(ls2)/len(ls2)