import numpy as np
import math

from IoTEnv import Uav, Env, Policy
from UAVEnergy import UAV_Energy
import Util

class RandomStrategy:
    def __init__(self, param, logging_timeline) -> None:
        self.param = param
        self.logging_timeline = logging_timeline
        self.devices = Util.initialize_fixed_devices(param)
        self.uav = Uav(param['V'], self.devices)
        self.env = Env(self.devices, self.uav, param['nTimeUnits_random'])
        self.ave_Reward_random = 0.0

    def learning(self):
        print("Random trajectory: One Episode Only")
        state_random = self.env.reset()
        ep_reward_random = 0
        t = 0
        n = 0  # logging fly behaviors
        Reward_random = []
        PV_random = []
        while t < self.param['nTimeUnits_random']:
            action_random = np.random.randint(self.param['num_Devices'])  # 纯粹随机选择
            CPoint = self.env.UAV.location  # current location
            NPoint = self.env.Devices[action_random].location  # next location
            distance = np.linalg.norm(CPoint - NPoint)  # Compute the distance of two points
            # env_random.UAV.V = logging_timeline[0][param['episodes']]['UAV_VelocityList'][-1]
            Fly_time = 1 if distance == 0 else math.ceil(distance / self.env.UAV.V)
            PV = UAV_Energy(self.param['V']) * Fly_time
            t = t + Fly_time
            if t > self.param['nTimeUnits_random']:
                break
            n = n + 1
            state_random, reward_, reward_rest, reward_random  = self.env.step(state_random, action_random, self.param['V'], t, PV, self.param, Fly_time)
            # model.rewards_random.append(reward_random)
            PV_random.append(PV)
            Reward_random.append(reward_random)
            ep_reward_random += reward_random
            # model.actions_random.append(action_random)
            # model.states_random.append(state_random)
            print("Random: The {} episode" " and the {} fly" " at the end of {} time slots. " "Visit device {}".format(1, n,
                                                                                                                       t,
                                                                                                                       action_random))

        self.logging_timeline[0][0]['Reward_random'] = Reward_random
        self.logging_timeline[0][0]['Random_UAV_PositionList'] = self.uav.PositionList
        self.logging_timeline[0][0]['Random_UAV_PositionCor'] = self.uav.PositionCor
        self.logging_timeline[0][0]['Random_UAV_VelocityList'] = self.uav.VelocityList
        self.logging_timeline[0][0]['Random_UAV_Reward'] = self.uav.Reward
        self.logging_timeline[0][0]['Random_UAV_Energy'] = self.uav.Energy
        self.logging_timeline[0][0]['Random_UAV_R_E'] = self.uav.Sum_R_E
        self.logging_timeline[0][0]['Random_UAV_AoI'] = self.uav.AoI
        self.logging_timeline[0][0]['Random_UAV_CPU'] = self.uav.CPU
        self.logging_timeline[0][0]['Random_UAV_b'] = self.uav.b
        for i in range(self.param['num_Devices']):
            self.logging_timeline[i][0]['Random_KeyTime'] = self.devices[i].KeyTime
            self.logging_timeline[i][0]['Random_TaskList'] = self.devices[i].TaskList
            # 记录每一个EPISODE的非REGULAR的数据
            self.logging_timeline[i][0]['Random_KeyRewards'] = self.devices[i].KeyReward
            ls2 = self.logging_timeline[i][0]['Random_KeyRewards']
            if len(self.logging_timeline[i][0]['Random_KeyTime']) == 1:
                self.logging_timeline[i][0]['Random_avg_reward'] = None
            else:
                self.logging_timeline[i][0]['Random_avg_reward'] = sum(ls2)/len(ls2)
        self.ave_Reward_random = ep_reward_random / n
        self.Ep_Reward_random = ep_reward_random
        print('Random: Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(1, ep_reward_random, self.ave_Reward_random))
