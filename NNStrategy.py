import math
import torch
import numpy as np
from collections import namedtuple

from IoTEnv import Uav, Env, Policy
from UAVEnergy import UAV_Energy
import Util

class NNStrategy:
    def __init__(self, param, logging_timeline) -> None:
        self.param = param
        self.logging_timeline = logging_timeline
        self.episode_reward = []
        self.average_reward = []
        self.devices = Util.initialize_fixed_devices(param)

        self.uav = Uav(param['V'], self.devices)
        self.env = Env(self.devices, self.uav, param['nTimeUnits'])

        self.model = Policy(1 * param['num_Devices'], param['num_Devices'] + 1, param['V_Lim'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=param['learning_rate'])  # lr=3e-2
        self.eps = np.finfo(np.float32).eps.item()

    def select_action(self, state):
        # state = torch.from_numpy(state).float()
        state = torch.from_numpy(state).double()
        probs, state_value, velocity = self.model(state)

        # create a categorical distribution over the list of probabilities of actions
        print(probs)
        print(state_value)
        print(velocity)
        m = torch.distributions.Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()
        # action = torch.tensor(2)

        # print(state)
        for label, p in enumerate(probs):
            print(f'{label:2}: {100 * p:5.2f}%')
        # print("---", action, "is chosen")

        # save to action buffer
        savedAction = namedtuple('SavedAction', ['log_prob', 'value', 'velocity'])
        self.model.saved_actions.append(savedAction(
            m.log_prob(action), state_value, velocity))

        # the action to take
        return action.item(), velocity.item()

    def finish_episode(self):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        R = 0
        saved_actions = self.model.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        # velocity_losses = []
        returns = []  # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.model.rewards[::-1]:
            # calculate the discounted value
            R = r + self.param['gamma'] * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value, velocity), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage - velocity * advantage)
            # policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(torch.nn.functional.smooth_l1_loss(value, torch.tensor([R])))

            # 尝试添加速度：没必要，因为上面计算policy_losses的时候，就已经有了velocity的部分
            # velocity_losses.append(F.smooth_l1_loss(velocity, torch.tensor([R])))


        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
               # + torch.stack(velocity_losses).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.model.rewards[:]
        # del model.reward_[:]
        del self.model.saved_actions[:]


    def learning(self):

        rate1 = []
        rate2 = []

        for i_episode in range(1, self.param['episodes'] + 1):

            state = self.env.reset(self.devices, self.uav)
            # print("the initial state: ", state)
            print("----------------------------------------------------------------------------")
            print("       ")

            self.model.states.append(state)
            ep_reward = 0
            t = 0
            n_fly = 0  # logging fly behaviors
            # FIXME: when the distance is large or the velocity is small, the Fly_time can be too large to surpass the nTimeUnits


            rate1.append([])
            rate2.append([])
            while t < self.param['nTimeUnits']:
            # for t in range(0, param['nTimeUnits']):

                # select action from policy
                print("----------------------------------------------------------------------------")
                print('state:', state)

                action, velocity = self.select_action(state)
                # random action
                # action = np.random.randint(param['num_Devices'])

                # compute the distance
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
                state, reward_, reward_rest, reward = self.env.step(state, action, velocity, t, PV, self.param, Fly_time)
                print("reward: {}, reward_rest: {}, reward_: {}".format(reward_, reward_rest, reward_))
                n_fly += 1

                rate1[-1].append(reward_ / reward)
                rate2[-1].append(reward_rest / reward)

                self.model.actions.append(action)
                self.model.states.append(state)

                self.model.rewards.append(reward)
                ep_reward += reward

                print("Smart: The {} episode" " and the {} fly" " at the end of {} time slots. " "Visit device {}".format(i_episode, n_fly, t, action))

                print("----------------------------------------------------------------------------")
                print("       ")


            """
            FX3
            不除以N，凸起变高
            """

            ave_Reward = ep_reward
            # ave_Reward = sum(model.rewards) / n

            # update cumulative reward
            # running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            # perform backprop
            self.finish_episode()

            self.episode_reward.append(ep_reward / n_fly) # this episode/average seems the contrary
            # Running_reward.append(running_reward)
            self.average_reward.append(ep_reward)


            # save results in logging
            self.save_logging(i_episode)

            # pdb.set_trace()


            print("----------------------------------------------------------------------------")
            print("The percentage to all the devices:")
            for x in range(self.param['num_Devices']):
                action_list = self.model.actions[(i_episode - 1) * self.param['nTimeUnits']::]
                p = len([ele for ele in action_list if ele == x]) / self.param['nTimeUnits']
                print(f'{x:2}: {100 * p:5.2f}%')


            if i_episode % self.param['log_interval'] == 0:
                print('Smart: Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                      i_episode, ep_reward, ave_Reward))
        
    def save_logging(self, episode):
        # for i in range(param['num_Devices']):
        #     logging_timeline[i][EP]['timeline'].append(logging_timeline[i][EP]['intervals'][0])
        #     for j in range(1, len(logging_timeline[i][EP]['intervals'])):
        #         logging_timeline[i][EP]['timeline'].append(logging_timeline[i][EP]['timeline'][j-1] + logging_timeline[i][EP]['intervals'][j])
        # for x in range(1, EP):
        self.logging_timeline[0][episode]['UAV_TimeList'] = self.uav.TimeList
        self.logging_timeline[0][episode]['UAV_PositionList'] = self.uav.PositionList
        self.logging_timeline[0][episode]['UAV_PositionCor'] = self.uav.PositionCor
        self.logging_timeline[0][episode]['UAV_VelocityList'] = self.uav.VelocityList
        self.logging_timeline[0][episode]['UAV_Reward'] = self.uav.Reward  # 设备的COST(AOI+CPU)，负数，绝对值越小越好
        self.logging_timeline[0][episode]['UAV_Energy'] = self.uav.Energy  # UAV的飞行能量，正数，绝对值越小越好
        self.logging_timeline[0][episode]['UAV_R_E'] = self.uav.Sum_R_E  # 上面两项（REWARD+ENERGY）的和，负数，绝对值越小越好（这个是STEP输出的最后一个REWARD，优化目标本标，优化的是每个EPISODE的均值：Ep_reward）
        self.logging_timeline[0][episode]['UAV_AoI'] = self.uav.AoI  # 设备的AOI，正数，绝对值越小越好
        self.logging_timeline[0][episode]['UAV_CPU'] = self.uav.CPU  # 设备的CPU，正数，绝对值越小越好
        self.logging_timeline[0][episode]['UAV_b'] = self.uav.b      # 设备的B，正数，绝对值越小越好
        for i in range(self.param['num_Devices']):
            self.logging_timeline[i][episode]['intervals'] = self.devices[i].intervals
            self.logging_timeline[i][episode]['TimeList'] = self.devices[i].TimeList
            self.logging_timeline[i][episode]['TaskList'] = self.devices[i].TaskList
            self.logging_timeline[i][episode]['KeyTime'] = self.devices[i].KeyTime
            # 记录每一个EPISODE的非REGULAR的数据
            # FIXME: 这里的KEYREWARD只包含了AOI+CPU，没有包含UAV的能耗PV
            # 这里每一个DEVICE只能这样啊，DEVICE不像UAV一样一下一下飞，DEVICE是每一个时隙的
            # 这里的KEYREWARD是上面step输出reward的一部分，不包括UAV的PV，减200的penalty，不包括reward_rest
            self.logging_timeline[i][episode]['KeyTsk'] = self.devices[i].KeyTsk
            self.logging_timeline[i][episode]['KeyPol'] = self.devices[i].KeyPol
            self.logging_timeline[i][episode]['KeyRewards'] = self.devices[i].KeyReward
            self.logging_timeline[i][episode]['KeyAoI'] = self.devices[i].KeyAoI
            self.logging_timeline[i][episode]['KeyCPU'] = self.devices[i].KeyCPU
            self.logging_timeline[i][episode]['Keyb'] = self.devices[i].Keyb
            # 记录对应的REGULAR的数据
            self.logging_timeline[i][episode]['KeyTsk_Regular'] = self.devices[i].KeyTsk_Regular
            self.logging_timeline[i][episode]['KeyPol_Regular'] = self.devices[i].KeyPol_Regular
            self.logging_timeline[i][episode]['KeyReward_Regular'] = self.devices[i].KeyReward_Regular
            self.logging_timeline[i][episode]['KeyAoI_Regular'] = self.devices[i].KeyAoI_Regular
            self.logging_timeline[i][episode]['KeyCPU_Regular'] = self.devices[i].KeyCPU_Regular
            self.logging_timeline[i][episode]['Keyb_Regular'] = self.devices[i].Keyb_Regular

            ls1 = [0] + self.logging_timeline[i][episode]['intervals']
            ls2 = self.logging_timeline[i][episode]['KeyRewards']
            # 这里的avg_reward知识单纯的每一个device的reward均值
            if len(self.logging_timeline[i][episode]['KeyTime']) == 1:
                self.logging_timeline[i][episode]['avg_reward'] = None
            else:
                self.logging_timeline[i][episode]['avg_reward'] = sum([x * y for x, y in zip(ls1, ls2)]) / self.logging_timeline[i][episode]['KeyTime'][-1]