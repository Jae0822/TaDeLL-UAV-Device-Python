import numpy as np
import copy
import pickle
import random
import math

import torch.nn as nn
import torch.nn.functional as F

from associations import pg_rl
from env import Task


# Prepare the model and parameters
with open('mu_sig.pkl', 'rb') as f:
    mu, sig = pickle.load(f)  # Load the mu and sig to extract normalized feature

with open('TaDeLL_result_k_2.pkl', 'rb') as f:
    means_pg, means_tadell, niter, TaDeLL_Model, tasks0, tasks, testing_tasks, testing_tasks_pg, testing_tasks_TaDeLL = pickle.load(f)



class Device(object):
    def __init__(self, frequency, cpu_capacity, field):

        # Static Attributes
        self.frequency = frequency
        self.cpu_capacity = cpu_capacity
        self.field = field
        self.location = field * np.random.random_sample((2, 1))
        # self.flag = False  # if the device is visited, then flag = True


    def gen_TimeTaskList(self, nTimeUnits):
        TimeList = np.zeros(nTimeUnits)
        TaskList = []
        task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
                    cpu_max=self.cpu_capacity, p=0.4 * np.random.random_sample() + 0.3, d=2, k=2)
        task.extract_feature(mu, sig)  # Normalize the plain_feature using mu, sig. self.feature will be updated.
        TaskList.append(task)  # This is the very first task at time slot 0

        mean = self.frequency
        t = int(np.random.normal(mean, mean / 10))
        # task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
        #                 cpu_max=self.cpu_capacity, p=0.4 * np.random.random_sample() + 0.3, d=2, k=2)
        # task.extract_feature(mu, sig)  # Normalize the plain_feature using mu, sig. self.feature will be updated.
        # TaskList.append(task)
        while t < nTimeUnits:
            TimeList[t] = 1
            task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
                        cpu_max=self.cpu_capacity, p=0.4 * np.random.random_sample() + 0.3, d=2, k=2)
            task.extract_feature(mu, sig)  # Normalize the plain_feature using mu, sig. self.feature will be updated.
            TaskList.append(task)
            # t = t + int(np.random.normal(mean, mean / 10))
            t = t + mean

        return TimeList, TaskList



class Uav(object):
    def __init__(self, V):
        # Static Attributes
        self.init_location = np.zeros((2,1))
        self.location = np.zeros((2,1))
        self.V = V   # m/s



class Env(object):
    def __init__(self, Devices, UAV, nTimeUnits):
        self.Devices = Devices  # 提供多一层方便，不是形参，每一处的变动都会反映在原始的Devices上。
        self.UAV = UAV
        self.nTimeUnits = nTimeUnits
        self.num_Devices = len(Devices)

    def seed(self):
        pass

    def reset(self, Devices, UAV):
        # Reset Devices and UAV
        UAV.location = UAV.init_location
        for i in range(len(Devices)):
            Devices[i].TimeList, Devices[i].TaskList  = Devices[i].gen_TimeTaskList(self.nTimeUnits)   # The list of time that indicates the arrival of a new task
            Devices[i].nTasks = len(Devices[i].TaskList)
            Devices[i].NewTaskArrival = np.where(Devices[i].TimeList)[0]  # The list of New task arrival time
            Devices[i].ta_dex = 0  # current task index
            Devices[i].task = Devices[i].TaskList[Devices[i].ta_dex]  # current task
            Devices[i].TaskList_Regular = copy.deepcopy(Devices[i].TaskList)  # For the comparison without warm start
            Devices[i].task_Regular = Devices[i].TaskList_Regular[Devices[i].ta_dex]   # current task for comparison without warm start

            Devices[i].KeyTime = [0]  # The list of key time at which the policy changes (1. UAV visits 2. new task arrival)
            Devices[i].KeyPol = [Devices[i].TaskList[0].init_policy]  # The list of policy at/after key time slot
            tsk0 = copy.deepcopy(Devices[i].TaskList[0])
            Devices[i].KeyTsk = [tsk0]
            Devices[i].flag = True  # To indicate the first visit
            Devices[i].rewards = []
            Devices[i].intervals = []
            Devices[i].rewards_Regular = []
            Devices[i].KeyPol_Regular = copy.deepcopy(Devices[i].KeyPol)  # The Key points for Regular learning without warm start
            Devices[i].KeyTsk_Regular = copy.deepcopy(Devices[i].KeyTsk)
        # Reset state
        state = np.concatenate((# [0 for x in range(self.num_Devices)],  # 1.当前节点总的已访问次数
                                # [0 for x in range(self.num_Devices)],  # 2.当前节点、当前任务的已访问次数（新任务归1或者0，否则+1。其他节点不变
                                [0 for x in range(self.num_Devices)],  # 3.距离上次访问每一个点的时间，初始都为0，当前节点归0，其他节点+1
                                # [1 for x in range(self.num_Devices)],  # 4.UAV处得知的，每一个节点是否有新任务。当前节点信息最准确  # 上次访问的时候是否是新任务(因为不知道其他节点当前的情况)（boolean）
                                np.concatenate(UAV.location)
                                ))         # 5.UAV的初始位置
        return state  # (1 * num_Devices,)

    def step(self, state_, action, t):

        state = copy.deepcopy(state_)

        # CPoint = self.UAV.location  # current location
        # NPoint = self.Devices[action].location  # next location
        # distance = np.linalg.norm(CPoint - NPoint)  # Compute the distance of two points
        # Fly_time = 1 if distance == 0 else math.ceil(distance / self.UAV.V)
        # t = t + Fly_time
        # t = t + 1

        self.UAV.location = self.Devices[action].location
        # if not self.Devices[action].flag:
            # self.Devices[action].flag = True



        # 1.当前节点总的已访问次数
        # state[action] += 1
        # 3.距离上次访问每一个点的时间，初始都为0，当前节点归0，其他节点+1
        for i in range(self.num_Devices):
            # state[i + 2 * self.num_Devices] += 1
            state[i] += 1
        # state[action + 2 * self.num_Devices] = 0  # 当前节点被访问，距离当前节点的上次访问时间为0 or 1
        state[action] = 0
        # 4.UAV处得知的，每一个节点是否有新任务。当前节点信息最准确 # 每一个节点上次访问的时候是否是新任务(因为不知道其他节点当前的情况)，#当前节点当前任务被访问，故归0。其他节点不知道
        # state[action + self.num_Devices] = 0
        # 5.UAV的初始位置
        state[-2:] = np.concatenate(self.UAV.location)


        device = self.Devices[action]
        if not device.KeyTime:
            Last_Visted_Time = 0
            index_start = 0
        else:
            Last_Visted_Time = device.KeyTime[-1]
            index_start = device.KeyTime.index(Last_Visted_Time)

        #  ------------------Update policy for the current device ------------------#
        if device.flag and (not np.any(device.TimeList[device.KeyTime[-1]: t])):
            # it's first time and there's no more new task arrival before this visit (still the very first/initial task)
            # state[action + 1 * self.num_Devices] = 1  # 2.当前节点、当前任务的已访问次数（新任务归1或者0，否则+1。其他节点不变

            device.flag = False
            device.KeyTime.append(t)

            # 1: Warm Start
            TaDeLL_Model.getDictPolicy_Single(device.task)
            device.KeyPol.append(device.task.policy)
            tsk0 = copy.deepcopy(device.task)
            device.KeyTsk.append(tsk0)

            # 2: Regular (Without Warm Start)
            pg_rl(device.task_Regular, 1)  # update the PG policy for one step
            device.KeyPol_Regular.append(device.task_Regular.policy)
            tsk0_Regular = copy.deepcopy(device.task_Regular)
            device.KeyTsk_Regular.append(tsk0_Regular)
        elif np.any(device.TimeList[device.KeyTime[-1]: t]):  # when there's a new task arrived
            # state[action + 1 * self.num_Devices] = 1  # 2.当前节点、当前任务的已访问次数（新任务归1或者0，否则+1。其他节点不变
            if device.flag:  # If it's the first time visit. But there's a second new task arrives
                device.flag = False
            device.ta_dex = device.ta_dex + 1
            if device.ta_dex > self.Devices[action].nTasks - 1:
                print("index out!")  # FXIME: This might come from the shortage of TaskList
                # FIXED: remember to provide a task for the very beginning, i.e. t=0 in Device.gen_TimeTaskList()
            device.task = device.TaskList[device.ta_dex]
            device.task_Regular = device.TaskList_Regular[device.ta_dex]

            # FIXME: What if there is more than one task? index can only find the first "1"
            # FIXED: The tasks in the middle can be directly ignored because they get nothing improved.
            # Can be reckoned as a random task as the last one
            ind = device.KeyTime[-1] + np.where(device.TimeList[device.KeyTime[-1]: t])[0][
                0]  # Find the first index of time that has a new task arrival
            device.KeyTime.append(ind)

            # 1: Warm Start
            device.KeyPol.append(device.task.init_policy)  # For the policy changes not from UAV's update
            tsk0 = copy.deepcopy(device.task)
            device.KeyTsk.append(tsk0)  # tsk0 with initial policy
            device.KeyTime.append(t)
            TaDeLL_Model.getDictPolicy_Single(device.task)  # initialize the warm start policy
            device.KeyPol.append(device.task.policy)
            tsk0 = copy.deepcopy(device.task)
            device.KeyTsk.append(tsk0)  # tsk0 with the improved policy

            # 2: Regular (Without Warm Start)
            device.KeyPol_Regular.append(device.task_Regular.init_policy)
            tsk0_Regular = copy.deepcopy(device.task_Regular)
            device.KeyTsk_Regular.append(tsk0_Regular)
            pg_rl(device.task_Regular, 1)
            device.KeyPol_Regular.append(device.task_Regular.policy)
            tsk0_Regular = copy.deepcopy(device.task_Regular)
            device.KeyTsk_Regular.append(tsk0_Regular)

            # elif device.TimeList[t] == 1:  # If the new task arrival encounters the UAV visit
        #     device.ta_dex = device.ta_dex + 1
        #     device.task = device.TaskList[device.ta_dex]
        #
        #     device.KeyTime.append(t)
        #     TaDeLL_Model.getDictPolicy_Single(device.task)  # initialize the warm start policy
        #     device.KeyPol.append(device.task.policy)
        #     tsk0 = copy.deepcopy(device.task)
        #     device.KeyTsk.append(tsk0)  # tsk0 with the improved policy
        else:  # when this task has got warm start policy before
            # state[action + 1 * self.num_Devices] += 1   # 2.当前节点、当前任务的已访问次数（新任务归1或者0，否则+1。其他节点不变
            # device.task = device.TaskList[device.ta_dex]
            device.KeyTime.append(t)

            # 1: Warm Start
            pg_rl(device.task, 1)  # update the PG policy for one step
            device.KeyPol.append(device.task.policy)
            tsk0 = copy.deepcopy(device.task)
            device.KeyTsk.append(tsk0)  # tsk0 with the improved policy

            # 2: Regular (Warm Start)
            pg_rl(device.task_Regular, 1)
            device.KeyPol_Regular.append(device.task_Regular.policy)
            tsk0_Regular = copy.deepcopy(device.task_Regular)
            device.KeyTsk_Regular.append(tsk0_Regular)


        # ------------------Update estimated reward for rest devices----------------------#
        # FIXME: Fly_Time or state[i] （两种模式哪一种更好）
        # FIXME: end of the nTimeUnits, 怎么处理剩下的一小部分或多或少的时间。
        reward_rest = 0
        for i in range(self.num_Devices):
            if i == action:
                pass
            else:
                device = self.Devices[i]
                if device.flag:                # if never been visited by UAV
                    # reward_rest += state[i] * (-5)  # FIXME: choose a proper constant
                    reward_rest += -5
                    # 在TIMEUNITS够(100)的情况下，这个数值从 -1 ～ -60 都不会影响reward_, reward_rest的比例, 太大的话就有风险了
                else:                             # if this device has been visited by UAV
                    reward_rest += device.rewards[-1] * state[i]  # should decrease with time
                    # reward_rest += state[i] * device.rewards[-1]

        reward_rest = reward_rest / (self.num_Devices - 1)


        device = self.Devices[action]
        index_end = device.KeyTime.index(device.KeyTime[-1])
        # Update Rewards lists
        # 1: Warm Start rewards history
        reward_ = 0
        for index in range(index_start, index_end):
            tsk = device.KeyTsk[index]
            alpha = device.KeyPol[index]
            device.intervals.append(device.KeyTime[index+1] - device.KeyTime[index])
            reward = tsk.get_value(
                alpha['theta'])  # Didn't use pg_rl() cause it has one step of update which I don't need here
            device.rewards.append(reward)
            reward_ += device.intervals[-1] * device.rewards[-1]
        # add other devices' reward into account

        reward_ = reward_ / (device.KeyTime[index_end] - device.KeyTime[index_start]) # not the same as  device.intervals[-1]
        reward_final = (reward_ + reward_rest)/2
        # FIXME: Compute UAV's energy consumption when flies from previous point to next point
        # reward_Fly_energy = reward_ +


        # # 2: Regular (Without Warm Start)
        # reward_Regular = 0
        # for index in range(index_start, index_end):
        #     tsk = device.KeyTsk_Regular[index]
        #     alpha = device.KeyPol_Regular[index]
        #     reward = tsk.get_value(alpha['theta'])  # Cause the pg_rl has one step of update which I don't need here
        #     device.rewards_Regular.append(reward)
        #     reward_Regular += device.intervals[-1] * device.rewards_Regular[-1]


        # print("done one step")
        # return state, reward_, reward_Regular, t
        return state, reward_, reward_rest, reward_final

    def update(self):
        pass



class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, 32)
        self.affine2 = nn.Linear(32, 64)
        self.affine3 = nn.Linear(64, 128)


        # actor's layer
        # self.action_affine1 = nn.Linear(32, 64)
        self.action_head = nn.Linear(128, output_size)

        # critic's layer
        # self.value_affine1 = nn.Linear(32, 64)
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.actions = []  # To record the actions
        self.states = []  # To record the states
        self.rewards = []
        self.reward_ = []
        self.reward_rest = []

        self.actions_random = []  # To record the actions
        self.states_random = []  # To record the states
        self.rewards_random = []
        self.double()

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        # a = F.relu(self.action_affine1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        # v = F.relu(self.value_affine1(x))
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values








