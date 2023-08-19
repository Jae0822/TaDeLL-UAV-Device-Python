import numpy as np
import math

from IoTEnv import Uav, Env, Policy
from UAVEnergy import UAV_Energy
import Util

class ForcedStrategy:
    def __init__(self, param, logging_timeline) -> None:
        self.param = param
        self.logging_timeline = logging_timeline
        self.devices = Util.initialize_fixed_devices(param)
        self.uav = Uav(param['V'], self.devices)
        self.env = Env(self.devices, self.uav, param['nTimeUnits_force'])
        self.ave_Reward_force = 0.0

    def learning(self):
        print("Forced trajectory: One Episode Only")
        state_force = self.env.reset(self.devices, self.uav)
        ep_reward_force = 0
        t = 0
        n = 0
        Reward_force = []
        PV_force = []
        while t < self.param['nTimeUnits_force']:

            # 强制选择action
            action_table_force = np.zeros(self.param['num_Devices'])  # 筛选出当前有新任务的设备
            for i in range(self.param['num_Devices']):
                if self.devices[i].TimeList[t-1] == 1:
                    action_table_force[i] = 1
            inx = np.where(action_table_force == 1)[0]
            # action = inx[np.random.randint(len(inx))] if inx else np.random.randint(param['num_Devices']) # 随机选一个去访问
            if inx.any():
                action_force = inx[0]  # inx[np.random.randint(len(inx))] #优先选择变化最不频繁的
            else:
                action_force = np.random.randint(self.param['num_Devices'])
            # compute the distance
            CPoint = self.env.UAV.location  # current location
            NPoint = self.env.Devices[action_force].location  # next location
            distance = np.linalg.norm(CPoint - NPoint)  # Compute the distance of two points

            Fly_time = 1 if distance == 0 else math.ceil(distance / self.env.UAV.V)
            PV = UAV_Energy(self.param['V']) * Fly_time
            t = t + Fly_time
            if t > self.param['nTimeUnits_force']:
                break
            n = n + 1
            state_force, reward_, reward_rest, reward_force = self.env.step(state_force, action_force, self.param['V'], t, PV, self.param, Fly_time)
            PV_force.append(PV)
            Reward_force.append(reward_force)
            ep_reward_force += reward_force
            print("Force: The {} episode" " and the {} fly" " at the end of {} time slots. " "Visit device {}".format(1, n,t,action_force))
        self.logging_timeline[0][0]['Reward_force'] = Reward_force
        self.logging_timeline[0][0]['Force_UAV_TimeList'] = self.uav.TimeList
        self.logging_timeline[0][0]['Force_UAV_PositionList'] = self.uav.PositionList
        self.logging_timeline[0][0]['Force_UAV_PositionCor'] = self.uav.PositionCor
        self.logging_timeline[0][0]['Force_UAV_VelocityList'] = self.uav.VelocityList
        self.logging_timeline[0][0]['Force_UAV_Reward'] = self.uav.Reward
        self.logging_timeline[0][0]['Force_UAV_Energy'] = self.uav.Energy
        self.logging_timeline[0][0]['Force_UAV_R_E'] = self.uav.Sum_R_E
        self.logging_timeline[0][0]['Force_UAV_AoI'] = self.uav.AoI
        self.logging_timeline[0][0]['Force_UAV_CPU'] = self.uav.CPU
        self.logging_timeline[0][0]['Force_UAV_b'] = self.uav.b
        for i in range(self.param['num_Devices']):
            self.logging_timeline[i][0]['Force_intervals'] = self.devices[i].intervals
            self.logging_timeline[i][0]['Force_TimeList'] = self.devices[i].TimeList
            self.logging_timeline[i][0]['Force_KeyTime'] = self.devices[i].KeyTime
            self.logging_timeline[i][0]['Force_TaskList'] = self.devices[i].TaskList
            # 记录每一个EPISODE的非REGULAR的数据
            self.logging_timeline[i][0]['Force_KeyTsk'] = self.devices[i].KeyTsk
            self.logging_timeline[i][0]['Force_KeyPol'] = self.devices[i].KeyPol
            self.logging_timeline[i][0]['Force_KeyRewards'] = self.devices[i].KeyReward
            self.logging_timeline[i][0]['Force_KeyAoI'] = self.devices[i].KeyAoI
            self.logging_timeline[i][0]['Force_KeyCPU'] = self.devices[i].KeyCPU
            self.logging_timeline[i][0]['Force_Keyb'] = self.devices[i].Keyb
            # 记录对应的REGULAR的数据
            self.logging_timeline[i][0]['Force_KeyTsk_Regular'] = self.devices[i].KeyTsk_Regular
            self.logging_timeline[i][0]['Force_KeyPol_Regular'] = self.devices[i].KeyPol_Regular
            self.logging_timeline[i][0]['Force_KeyReward_Regular'] = self.devices[i].KeyReward_Regular
            self.logging_timeline[i][0]['Force_KeyAoI_Regular'] = self.devices[i].KeyAoI_Regular
            self.logging_timeline[i][0]['Force_KeyCPU_Regular'] = self.devices[i].KeyCPU_Regular
            self.logging_timeline[i][0]['Force_Keyb_Regular'] = self.devices[i].Keyb_Regular
            ls1 = [0] + self.logging_timeline[i][0]['Force_intervals']
            ls2 = self.logging_timeline[i][0]['Force_KeyRewards']
            if len(self.logging_timeline[i][0]['Force_KeyTime']) == 1:
                self.logging_timeline[i][0]['Force_avg_reward'] = None
            else:
                self.logging_timeline[i][0]['Force_avg_reward'] = sum([x * y for x, y in zip(ls1, ls2)]) / \
                                                       self.logging_timeline[i][0]['Force_KeyTime'][-1]
        self.ave_Reward_force = ep_reward_force / n
        print('Force: Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(1, ep_reward_force, self.ave_Reward_force))