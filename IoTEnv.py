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
    def __init__(self, frequency, cpu_capacity, field, taskBase):

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

        self.taskBase = taskBase

        self.KeyTime = [0]  # The list of key time at which the policy changes (1. UAV visits 2. new task arrival)
        self.KeyReward = [0]  # Didn't use pg_rl() cause it has one step of update which I don't need here
        self.KeyAoI = [0]
        self.KeyCPU = [0]
        self.Keyb = [0]
        #self.KeyReward = [tsk0.get_value(tsk0.init_policy['theta'])]  # Didn't use pg_rl() cause it has one step of update which I don't need here

        self.TaskList = []
        self.missedTasks = {}
        self.lastUavVisit = -1
        self.lastMissedTask = 0


    def gen_TimeTaskList_set(self, nTimeUnits):
        TaskList = [None]*nTimeUnits
        mean = self.frequency
        t = int(np.random.normal(mean, mean / 10))
        TaskList[0] = copy.deepcopy(self.taskBase)
        while t < nTimeUnits:
            TaskList[t] = copy.deepcopy(self.taskBase)
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

class TaskCache():
    def __init__(self, task):
        self.task = copy.deepcopy(task)
        self.value = []
        self.AoI_CPU = []
        self.idx = 0

    def populate(self, model):
        print("Building TaskCache")
        self.value.append(self.task.get_value(self.task.init_policy['theta']))
        self.AoI_CPU.append(self.task.get_AoI_CPU(self.task.init_policy['theta']))

        for i in range(1, 25):
            if model:
                model.getDictPolicy_Single(self.task)
            else:
                pg_rl(self.task, 1)
        
            self.value.append(self.task.get_value(self.task.policy['theta']))
            self.AoI_CPU.append(self.task.get_AoI_CPU(self.task.policy['theta']))
        print("Building TaskCache DONE, model: {} value {} AoI {}".format(model, self.value, self.AoI_CPU))
    
    def get_value(self):
        return self.value[1]

    def get_AoI_CPU(self):
        return self.AoI_CPU[self.idx]

    def visit(self):
        if self.idx < len(self.value) - 1:
            self.idx += 1


class Env(object):
    def __init__(self, Devices, UAV, nTimeUnits):
        self.Devices = Devices  # 提供多一层方便，不是形参，每一处的变动都会反映在原始的Devices上。
        self.UAV = UAV
        self.nTimeUnits = nTimeUnits
        self.num_Devices = len(Devices)

        self.initialization(self.Devices, self.UAV)


    def seed(self):
        pass

    def initialization(self, Devices, UAV):
        # initialize for each device
        for i in range(len(Devices)):
            Devices[i].TaskList_original = Devices[i].gen_TimeTaskList_set(self.nTimeUnits)   # The list of time that indicates the arrival of a new task

    def reset(self):
        # Reset Devices and UAV
        self.UAV.TimeList = [0]
        self.UAV.location = self.UAV.init_location
        self.UAV.PositionCor = [self.UAV.init_location]
        self.UAV.PositionList = [0]
        self.UAV.VelocityList = []
        self.UAV.Reward = []
        self.UAV.Energy = []
        self.UAV.Sum_R_E = []
        self.UAV.AoI = []
        self.UAV.CPU = []
        self.UAV.b = []
        for i in range(len(self.Devices)):
            self.Devices[i].TaskList = copy.deepcopy(self.Devices[i].TaskList_original)
            self.Devices[i].KeyTime = [0]  # The list of key time at which the policy changes (1. UAV visits 2. new task arrival)
            #self.KeyReward = [tsk0.get_value(tsk0.init_policy['theta'])]  # Didn't use pg_rl() cause it has one step of update which I don't need here
            self.Devices[i].KeyReward = [0]  # Didn't use pg_rl() cause it has one step of update which I don't need here
            self.Devices[i].KeyAoI = [0]
            self.Devices[i].KeyCPU = [0]
            self.Devices[i].Keyb = [0]
            self.Devices[i].missedTasks.clear()
            self.Devices[i].lastUavVisit = -1
            self.Devices[i].lastMissedTask = 0
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
        reward_, AoI_, CPU_, b_ = self.calculate_reward_since_last_visit(cur_device, t)
        cur_device.KeyAoI.append(AoI_)
        cur_device.KeyCPU.append(CPU_)
        cur_device.Keyb.append(b_)

        for dev in self.Devices:

            if (dev == cur_device):
                continue

            r, a, c, b = self.calculate_penalty_since_last_visit(dev, t)
            reward_rest += r
            dev.KeyAoI.append(a)
            dev.KeyCPU.append(c)
            dev.Keyb.append(b)

        reward_ = reward_ + reward_rest/(len(self.Devices) - 1)
        cur_device.KeyTime.append(t)
        cur_device.KeyReward.append(reward_)   # should decrease with time

        mu = param['mu']
        reward_fair = (1 - mu) * reward_ - mu * PV  # 添加飞行能量消耗
        print('reward_: ', reward_, ', Energy:', PV, 'reward fair: ', reward_fair)

        self.UAV.Reward.append(reward_)
        self.UAV.Energy.append(PV)
        self.UAV.Sum_R_E.append(reward_fair)
        self.UAV.AoI.append(AoI_)
        self.UAV.CPU.append(CPU_)
        self.UAV.b.append(b_)

        cur_device.missedTasks.clear()

        return state, reward_, reward_rest, reward_fair

    def calculate_reward_since_last_visit(self, device, time):
        reward = 0
        AoI = 0
        CPU = 0
        b = 0

        #for i in [i for i in range(len(test_list)) if test_list[i] != None]
        for i in range(device.lastUavVisit + 1, min(time+1, len(device.TaskList))):
            if device.TaskList[i] == None:
                continue
            interval = i - max(device.lastUavVisit, device.lastMissedTask)
            dataTask = device.TaskList[device.lastMissedTask].get_AoI_CPU()
            AoI += dataTask[0] * interval
            CPU += dataTask[1] * interval
            b += dataTask[2] * interval

            device.missedTasks[i] = device.TaskList[i].get_value() + 10
            device.lastMissedTask = i
        
        i = device.lastMissedTask
        interval = time - max(device.lastUavVisit, device.lastMissedTask)
        dataTask = device.TaskList[device.lastMissedTask].get_AoI_CPU()
        AoI += dataTask[0] * interval
        CPU += dataTask[1] * interval
        b += dataTask[2] * interval

        device.TaskList[i].visit()

        #improv_val = device.TaskList[i].get_value()
        #device.missedTasks.setdefault(i, 0)
        #device.missedTasks[i] += (improv_val + 10)
        #print("Improving reward by {} total {}".format(improv_val, device.missedTasks[i]))

        for t, cur_task in device.missedTasks.items():
            reward += cur_task
            #reward += cur_task*(max(0,(device.frequency - (time - t)))/device.frequency)

        device.lastUavVisit = time
        return reward, AoI, CPU, b
    
    def calculate_penalty_since_last_visit(self, device, time):
        reward = 0
        AoI = 0
        CPU = 0
        b = 0

        last_visited_time = device.KeyTime[-1]
        if (last_visited_time == 0):
            reward = -10
        
        #for i in [i for i in range(len(test_list)) if test_list[i] != None]
        for i in range(device.lastUavVisit + 1, min(time+1, len(device.TaskList))):
            if device.TaskList[i] == None:
                continue

            interval = i - max(device.lastUavVisit, device.lastMissedTask)
            dataTask = device.TaskList[device.lastMissedTask].get_AoI_CPU()
            AoI += dataTask[0] * interval
            CPU += dataTask[1] * interval
            b += dataTask[2] * interval

            device.missedTasks[i] = device.TaskList[i].get_value() + 10
            device.lastMissedTask = i
            #cur_task = device.TaskList[i]
            #interval = time - i
            #reward += cur_task.get_value(cur_task.init_policy['theta']) * interval
            #AoI += cur_task.get_AoI_CPU(cur_task.init_policy['theta'])[0] * interval
            #CPU += cur_task.get_AoI_CPU(cur_task.init_policy['theta'])[1] * interval
            #b += cur_task.get_AoI_CPU(cur_task.init_policy['theta'])[2] * interval
            #print("cur time {} index {} updating reward: {} inteval: {}".format(
            #    time, i, cur_task.get_value(cur_task.init_policy['theta']), interval))

        interval = time - max(device.lastUavVisit, device.lastMissedTask)
        dataTask = device.TaskList[device.lastMissedTask].get_AoI_CPU()
        AoI += dataTask[0] * interval
        CPU += dataTask[1] * interval
        b += dataTask[2] * interval

        for t, cur_task in device.missedTasks.items():
            interval = time - t
            reward += -cur_task
            #reward += cur_task * interval #FIXME
            #AoI += cur_task.get_AoI_CPU(cur_task.init_policy['theta'])[0] * interval
            #CPU += cur_task.get_AoI_CPU(cur_task.init_policy['theta'])[1] * interval
            #b += cur_task.get_AoI_CPU(cur_task.init_policy['theta'])[2] * interval

        device.lastUavVisit = time
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
        x = torch.nn.functional.normalize(torch.tensor(x, dtype=float), dim=0)
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

class ActorPolicy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, input_size, output_size, velocity):

        super(ActorPolicy, self).__init__()
        self.affine1 = nn.Linear(input_size, 64)
        self.pattern = [64]
        self.velocity = velocity

        # actor's layer
        if velocity != -1:
            output_size += 1

        self.action_head = nn.Linear(64, output_size)

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
        x = torch.nn.functional.normalize(torch.tensor(x, dtype=float), dim=0)
        x = F.relu(self.affine1(x))
        # actor: choses action to take from state s_t
        # by returning probability of each action
        # a = F.relu(self.action_affine1(x))
        x = self.action_head(x)
        action_prob = F.softmax(x[:-1], dim=-1)
        velocity = self.velocity
        if (self.velocity != -1):
            velocity = torch.clamp(x[-1], 0, 1)
            velocity = 5 + (self.velocity - 5)*velocity
        print("action prob {} velocity {}".format(action_prob, x[-1]))

        return action_prob, velocity

class CriticPolicy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, input_size):

        super(CriticPolicy, self).__init__()
        self.affine1 = nn.Linear(input_size, 64)
        # self.affine3 = nn.Linear(64, 128)
        # self.pattern = [32, 64, 128]
        self.pattern = [64]

        # critic's layer
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
        x = torch.nn.functional.normalize(torch.tensor(x, dtype=float), dim=0)
        x = F.relu(self.affine1(x))

        state_values = self.value_head(x)
        print("action value {}".format(state_values))

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return state_values