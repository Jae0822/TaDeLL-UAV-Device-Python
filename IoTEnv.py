import numpy as np
import copy
import pickle
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from associations import pg_rl
from env import Task

class Device(object):
    def __init__(self, frequency, cpu_capacity, field):

        # Static Attributes
        self.frequency = frequency
        self.cpu_capacity = cpu_capacity
        self.field = field
        self.location = field * np.random.random_sample((2, 1))  # results are from the “continuous uniform” distribution over the stated interval.
        # Prepare the model and parameters
        with open('input_files/mu_sig.pkl', 'rb') as f:
            self.mu, self.sig = pickle.load(f)  # Load the mu and sig to extract normalized feature

        with open('input_files/SourceTask_temp.pkl', 'rb') as f:
            self.taskList_set, _ = pickle.load(f)


    def gen_TimeTaskList_set(self, nTimeUnits):
        TaskList = [None]*nTimeUnits
        mean = self.frequency
        t = int(np.random.normal(mean, mean / 10))
        TaskList[0] = self.taskList_set[1]
        while t < nTimeUnits:
            TaskList[t] = self.taskList_set[1]
            t = t + mean
        return TaskList

class Uav(object):
    def __init__(self, V, Devices):
        # Static Attributes
        # self.init_location = np.zeros((2,1))
        self.init_location = Devices[0].location
        self.location = np.zeros((2,1))
        self.V = V   # m/s
        self.TimeList = [0]
        self.PositionCor = [self.init_location]  # list of sequent locations in an episode
        self.PositionList = [0]
        self.VelocityList = []
        self.Reward = []
        self.Energy = []
        self.Sum_R_E = []
        self.AoI = []
        self.CPU = []



class Env(object):
    def __init__(self, Devices, UAV, nTimeUnits):
        self.Devices = Devices  # 提供多一层方便，不是形参，每一处的变动都会反映在原始的Devices上。
        self.UAV = UAV
        self.nTimeUnits = nTimeUnits
        self.num_Devices = len(Devices)

        with open('input_files/TaDeLL_result_k_2.pkl', 'rb') as f:
            _, _, _, self.TaDeLL_Model, _, _, _, _, _ = pickle.load(f)

        self.initialization(Devices, UAV)


    def seed(self):
        pass

    def initialization(self, Devices, UAV):
        # initialize for each device
        for i in range(len(Devices)):
            Devices[i].TaskList = Devices[i].gen_TimeTaskList_set(self.nTimeUnits)   # The list of time that indicates the arrival of a new task

    def reset(self, Devices, UAV):
        # Reset Devices and UAV
        UAV.TimeList = [0]
        UAV.location = UAV.init_location
        UAV.PositionCor = [UAV.init_location]
        UAV.PositionList = [0]
        UAV.VelocityList = []
        UAV.Reward = []
        UAV.Energy = []
        UAV.Sum_R_E = []
        UAV.AoI = []
        UAV.CPU = []
        UAV.b = []
        for i in range(len(Devices)):
            Devices[i].ta_dex = 0  # current task index
            Devices[i].task = Devices[i].TaskList[Devices[i].ta_dex]  # current task
            Devices[i].TaskList_Regular = copy.deepcopy(Devices[i].TaskList)  # For the comparison without warm start

            Devices[i].KeyTime = [0]  # The list of key time at which the policy changes (1. UAV visits 2. new task arrival)
            Devices[i].KeyPol = [Devices[i].TaskList[0].init_policy]  # The list of policy at/after key time slot
            tsk0 = copy.deepcopy(Devices[i].TaskList[0])
            Devices[i].KeyTsk = [tsk0]
            Devices[i].KeyReward = [tsk0.get_value(tsk0.init_policy['theta'])]  # Didn't use pg_rl() cause it has one step of update which I don't need here
            Devices[i].KeyAoI = [tsk0.get_AoI_CPU(tsk0.init_policy['theta'])[0]]
            Devices[i].KeyCPU = [tsk0.get_AoI_CPU(tsk0.init_policy['theta'])[1]]
            Devices[i].Keyb = [tsk0.get_AoI_CPU(tsk0.init_policy['theta'])[2]]
            Devices[i].nonvisitFlag = True  # To indicate the first visit. This device hasn't been visited.
            Devices[i].rewards = []
            Devices[i].intervals = []
            Devices[i].rewards_Regular = []
            Devices[i].KeyPol_Regular = copy.deepcopy(Devices[i].KeyPol)  # The Key points for Regular learning without warm start
            Devices[i].KeyTsk_Regular = copy.deepcopy(Devices[i].KeyTsk)
            Devices[i].KeyReward_Regular = copy.deepcopy(Devices[i].KeyReward)
            Devices[i].KeyAoI_Regular = copy.deepcopy(Devices[i].KeyAoI)
            Devices[i].KeyCPU_Regular = copy.deepcopy(Devices[i].KeyCPU)
            Devices[i].Keyb_Regular = copy.deepcopy(Devices[i].Keyb)
        # Reset state
        state = np.concatenate((# [0 for x in range(self.num_Devices)],  # 1.当前节点总的已访问次数  不是已访问次数
                                # [0 for x in range(self.num_Devices)],  # 2.当前节点、当前任务的已访问次数（新任务归1或者0，否则+1。其他节点不变
                                # [0 for x in range(self.num_Devices)],    # 3.距离上次访问每一个点的时间，初始都为0，当前节点归0，其他节点+1
                                # [1 for x in range(self.num_Devices)],  # 4.UAV处得知的，每一个节点是否有新任务。当前节点信息最准确  # 上次访问的时候是否是新任务(因为不知道其他节点当前的情况)（boolean）
                                [0 for x in range(self.num_Devices)],    # 5. 当前任务出现时间长短（需要考虑飞行时间，也存在估计可能），正常情况+1，遇到新任务归0
                                # np.concatenate(UAV.location)            # UAV的初始位置
                                ))
        return state  # (1 * num_Devices,)

    def step(self, state, action, velocity, t, PV, param, Fly_time):
        # update the current state based on action/velocity/etc
        # device: update key time(uav visited or env change)

        self.UAV.location = self.Devices[action].location # new location for UAV
        self.UAV.TimeList.append(t) #list with times that UAV arrived at locations
        self.UAV.PositionCor.append(self.Devices[action].location) # list positions
        self.UAV.PositionList.append(action) # list of nodes UAV visited
        self.UAV.VelocityList.append(velocity) # list velocity

        for i in range(self.num_Devices):
            state[i] = state[i] + Fly_time # 当前flying time为1
        state[action] = 0

        cur_device = self.Devices[action]
        #index_start = self.update_device_since_keytime(cur_device, t)

        # update reward
        reward_rest = 0
        reward_ = 0
        AoI_ = 0
        CPU_ = 0
        b_  = 0
        for dev in self.Devices:
            r, a, c, b = self.calculate_reward_since_last_visit(dev, t)
            reward_ += r
            AoI_ += a
            CPU_ += c
            b_ += b

        cur_device.KeyTime.append(t)
        cur_device.KeyReward.append(reward_)   # should decrease with time

        reward_final = reward_  # weighted average,  iteration :discounted reward, back discount
        alpha = param['alpha']
        reward_fair = reward_final

        mu = param['mu']
        reward_fair1 = (1 - mu) * reward_fair - mu * PV  # 添加飞行能量消耗
        print('reward: ', reward_fair, ', Energy:', PV)

        self.UAV.Reward.append(reward_fair)
        self.UAV.Energy.append(PV)
        self.UAV.Sum_R_E.append(reward_fair1)
        self.UAV.AoI.append(AoI_)
        self.UAV.CPU.append(CPU_)
        self.UAV.b.append(b_)

        return state, reward_, reward_rest, reward_fair1

    def calculate_reward_since_last_visit(self, device, time):
        reward = 0
        AoI = 0
        CPU = 0
        b = 0

        last_visited_time = device.KeyTime[-1]
        #for i in [i for i in range(len(test_list)) if test_list[i] != None]
        for i in np.nonzero(device.TaskList)[0]:
            if i > time:
                break
            if i <= last_visited_time:
                continue
            cur_task = device.TaskList[i]
            interval = time - i
            reward += cur_task.get_value(cur_task.init_policy['theta']) * interval
            AoI += cur_task.get_AoI_CPU(cur_task.init_policy['theta'])[0] * interval
            CPU += cur_task.get_AoI_CPU(cur_task.init_policy['theta'])[1] * interval
            b += cur_task.get_AoI_CPU(cur_task.init_policy['theta'])[2] * interval
            #print("cur time {} index {} updating reward: {} inteval: {}".format(
            #    time, i, cur_task.get_value(cur_task.init_policy['theta']), interval))
        return reward, AoI, CPU, b

    def update(self):
        pass

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, input_size, output_size, velocity_lim):

        self.velocity_lim = velocity_lim

        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, 32)
        self.affine2 = nn.Linear(32, 64)
        # self.affine3 = nn.Linear(64, 128)
        # self.pattern = [32, 64, 128]
        self.pattern = [32, 64]


        # actor's layer
        # self.action_affine1 = nn.Linear(32, 64)
        self.action_head = nn.Linear(64, output_size)
        self.velocity_head = nn.Linear(64, 1)
        # self.velocity_head.weight.requires_grad_(False) # 禁用梯度

        # critic's layer
        # self.value_affine1 = nn.Linear(32, 64)
        self.value_head = nn.Linear(64, 1)

        # action & reward buffer
        self.saved_actions = []
        self.actions = []  # To record the actions
        self.states = []  # To record the states
        self.rewards = []

        self.double()

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        # x = F.relu(self.affine3(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        # a = F.relu(self.action_affine1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        print("action prob {}".format(self.action_head(x)))
        print("action value {}".format(self.value_head(x)))
        velocity = torch.clamp(self.velocity_head(x), 5, self.velocity_lim)
        print('velocity:', velocity)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values, velocity
