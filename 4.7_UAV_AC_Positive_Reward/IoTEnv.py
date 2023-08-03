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
from UAVEnergy import UAV_Energy

# Prepare the model and parameters
with open('mu_sig.pkl', 'rb') as f:
    mu, sig = pickle.load(f)  # Load the mu and sig to extract normalized feature

with open('TaDeLL_result_k_2.pkl', 'rb') as f:
    means_pg, means_tadell, niter, TaDeLL_Model, tasks0, tasks, testing_tasks, testing_tasks_pg, testing_tasks_TaDeLL = pickle.load(f)

with open('SourceTask_temp.pkl', 'rb') as f:
    TaskList_set, Values_array_set = pickle.load(f)


class Device(object):
    def __init__(self, frequency, cpu_capacity, field):

        # Static Attributes
        self.frequency = frequency
        self.cpu_capacity = cpu_capacity
        self.field = field
        self.location = field * np.random.random_sample((2, 1))  # results are from the “continuous uniform” distribution over the stated interval.
        self.TimeSinceLastVisit = 0
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

    def gen_TimeTaskList_set(self, nTimeUnits):
        TimeList = np.zeros(nTimeUnits)
        TaskList = []
        TaskList.append(TaskList_set[1])
        mean = self.frequency
        t = int(np.random.normal(mean, mean / 10))
        while t < nTimeUnits:
            TimeList[t] = 1
            TaskList.append(TaskList_set[1])
            # t = t + int(np.random.normal(mean, mean / 10))
            t = t + mean
        return TimeList, TaskList

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

    def seed(self):
        pass

    def initialization(self, Devices, UAV):
        # initialize for each device
        for i in range(len(Devices)):
            Devices[i].TimeList, Devices[i].TaskList  = Devices[i].gen_TimeTaskList_set(self.nTimeUnits)   # The list of time that indicates the arrival of a new task
            Devices[i].nTasks = len(Devices[i].TaskList)
            Devices[i].NewTaskArrival = np.where(Devices[i].TimeList)[0]  # The list of New task arrival time

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
            # Devices[i].TimeList, Devices[i].TaskList  = Devices[i].gen_TimeTaskList(self.nTimeUnits)   # The list of time that indicates the arrival of a new task
            # Devices[i].nTasks = len(Devices[i].TaskList)
            # Devices[i].NewTaskArrival = np.where(Devices[i].TimeList)[0]  # The list of New task arrival time
            Devices[i].TimeSinceLastVisit = 0
            Devices[i].ta_dex = 0  # current task index
            Devices[i].task = Devices[i].TaskList[Devices[i].ta_dex]  # current task
            Devices[i].TaskList_Regular = copy.deepcopy(Devices[i].TaskList)  # For the comparison without warm start
            Devices[i].task_Regular = Devices[i].TaskList_Regular[Devices[i].ta_dex]   # current task for comparison without warm start

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

    def step(self, state_, action, velocity, t, PV, param):

        state = copy.deepcopy(state_)




        # CPoint = self.UAV.location  # current location
        # NPoint = self.Devices[action].location  # next location
        # distance = np.linalg.norm(CPoint - NPoint)  # Compute the distance of two points
        # Fly_time = 1 if distance == 0 else math.ceil(distance / self.UAV.V)
        # t = t + Fly_time
        # t = t + 1

        self.UAV.location = self.Devices[action].location
        self.UAV.TimeList.append(t)
        self.UAV.PositionCor.append(self.Devices[action].location)
        self.UAV.PositionList.append(action)
        self.UAV.VelocityList.append(velocity)
        # if not self.Devices[action].flag:
            # self.Devices[action].flag = True

        for i in range(self.num_Devices):
            self.Devices[i].TimeSinceLastVisit += 1
        self.Devices[action].TimeSinceLastVisit = 0

        # 1.当前节点总的已访问次数
        # state[action] += 1
        # 3.距离上次访问每一个点的时间，初始都为0，当前节点归0，其他节点+1
        # for i in range(self.num_Devices):
        #     # state[i + 2 * self.num_Devices] += 1
        #     state[i] += 1
        # # state[action + 2 * self.num_Devices] = 0  # 当前节点被访问，距离当前节点的上次访问时间为0 or 1
        # state[action] = 0
        # 4.UAV处得知的，每一个节点是否有新任务。当前节点信息最准确 # 每一个节点上次访问的时候是否是新任务(因为不知道其他节点当前的情况)，#当前节点当前任务被访问，故归0。其他节点不知道
        # state[action + self.num_Devices] = 0
        # 5. 当前任务出现时间长短（需要考虑飞行时间，也存在估计可能），正常情况+1，遇到新任务归0
        # FIXME: 这里的不对，加的FLY TIME要改，不再是1了！！！！
        for i in range(self.num_Devices):  # 对所有的device，包括当前。
            state[i] += 1 # 当前flying time为1
        # UAV的初始位置
        # state[-2:] = np.concatenate(self.UAV.location)


        device = self.Devices[action]
        device.nonvisitFlag = False
        if not device.KeyTime:
            Last_Visted_Time = 0
            index_start = 0
        else:
            Last_Visted_Time = device.KeyTime[-1]
            index_start = device.KeyTime.index(Last_Visted_Time)



        VisitTime = device.TimeList[device.KeyTime[-1]+1 : t+1]

        #
        # #  ------------------Update policy for the current device ------------------#
        # if device.flag and (not np.any(device.TimeList[device.KeyTime[-1]: t])):
        #     # it's first time and there's no more new task arrival before this visit (still the very first/initial task)
        #     # state[action + 1 * self.num_Devices] = 1  # 2.当前节点、当前任务的已访问次数（新任务归1或者0，否则+1。其他节点不变
        #
        #     device.flag = False
        #     device.KeyTime.append(t)
        #
        #     # 1: Warm Start
        #     TaDeLL_Model.getDictPolicy_Single(device.task)
        #     device.KeyPol.append(device.task.policy)
        #     tsk0 = copy.deepcopy(device.task)
        #     device.KeyTsk.append(tsk0)
        #
        #     # 2: Regular (Without Warm Start)
        #     pg_rl(device.task_Regular, 1)  # update the PG policy for one step
        #     device.KeyPol_Regular.append(device.task_Regular.policy)
        #     tsk0_Regular = copy.deepcopy(device.task_Regular)
        #     device.KeyTsk_Regular.append(tsk0_Regular)
        # elif np.any(device.TimeList[device.KeyTime[-1] + 1: t]):  # when there's a new task arrived
        #     # state[action + 1 * self.num_Devices] = 1  # 2.当前节点、当前任务的已访问次数（新任务归1或者0，否则+1。其他节点不变
        #     if device.flag:  # If it's the first time visit. But there's a second new task arrives
        #         device.flag = False
        #     device.ta_dex = device.ta_dex + 1
        #     if device.ta_dex > self.Devices[action].nTasks - 1:
        #         print("index out!")  # FXIME: This might come from the shortage of TaskList
        #         # FIXED: remember to provide a task for the very beginning, i.e. t=0 in Device.gen_TimeTaskList()
        #     device.task = device.TaskList[device.ta_dex]
        #     device.task_Regular = device.TaskList_Regular[device.ta_dex]
        #
        #     # FIXME: What if there is more than one task? index can only find the first "1"
        #     # FIXED: The tasks in the middle can be directly ignored because they get nothing improved.
        #     # Can be reckoned as a random task as the last one
        #     ind = device.KeyTime[-1] + 1 + np.where(device.TimeList[device.KeyTime[-1] + 1: t])[0][
        #         0]  # Find the first index of time that has a new task arrival
        #     if ind == device.KeyTime[-1]:
        #         print('bug appears!')
        #     device.KeyTime.append(ind)
        #     state[action] = t - device.KeyTime[-1]
        #
        #     # 1: Warm Start
        #     device.KeyPol.append(device.task.init_policy)  # For the policy changes not from UAV's update
        #     tsk0 = copy.deepcopy(device.task)
        #     device.KeyTsk.append(tsk0)  # tsk0 with initial policy
        #     device.KeyTime.append(t)
        #     TaDeLL_Model.getDictPolicy_Single(device.task)  # initialize the warm start policy
        #     device.KeyPol.append(device.task.policy)
        #     tsk0 = copy.deepcopy(device.task)
        #     device.KeyTsk.append(tsk0)  # tsk0 with the improved policy
        #
        #     # 2: Regular (Without Warm Start)
        #     device.KeyPol_Regular.append(device.task_Regular.init_policy)
        #     tsk0_Regular = copy.deepcopy(device.task_Regular)
        #     device.KeyTsk_Regular.append(tsk0_Regular)
        #     pg_rl(device.task_Regular, 1)
        #     device.KeyPol_Regular.append(device.task_Regular.policy)
        #     tsk0_Regular = copy.deepcopy(device.task_Regular)
        #     device.KeyTsk_Regular.append(tsk0_Regular)
        #
        #     # elif device.TimeList[t] == 1:  # If the new task arrival encounters the UAV visit
        # #     device.ta_dex = device.ta_dex + 1
        # #     device.task = device.TaskList[device.ta_dex]
        # #
        # #     device.KeyTime.append(t)
        # #     TaDeLL_Model.getDictPolicy_Single(device.task)  # initialize the warm start policy
        # #     device.KeyPol.append(device.task.policy)
        # #     tsk0 = copy.deepcopy(device.task)
        # #     device.KeyTsk.append(tsk0)  # tsk0 with the improved policy
        # else:  # when this task has got warm start policy before
        #     # state[action + 1 * self.num_Devices] += 1   # 2.当前节点、当前任务的已访问次数（新任务归1或者0，否则+1。其他节点不变
        #     # device.task = device.TaskList[device.ta_dex]
        #     device.KeyTime.append(t)
        #
        #     # 1: Warm Start
        #     pg_rl(device.task, 1)  # update the PG policy for one step
        #     device.KeyPol.append(device.task.policy)
        #     tsk0 = copy.deepcopy(device.task)
        #     device.KeyTsk.append(tsk0)  # tsk0 with the improved policy
        #
        #     # 2: Regular (Warm Start)
        #     pg_rl(device.task_Regular, 1)
        #     device.KeyPol_Regular.append(device.task_Regular.policy)
        #     tsk0_Regular = copy.deepcopy(device.task_Regular)
        #     device.KeyTsk_Regular.append(tsk0_Regular)
        #










        # ------------------Update estimated reward for rest devices----------------------#
        # FIXME: Fly_Time or state[i] （两种模式哪一种更好）
        # FIXME: end of the nTimeUnits, 怎么处理剩下的一小部分或多或少的时间。

        if not np.any(VisitTime):
            "两次访问之间为全0，没有任何新任务出现"
            device.KeyTime.append(t)
            # 1: Warm Start
            if device.task.nonvisitFlag:
                TaDeLL_Model.getDictPolicy_Single(device.task)
                device.task.nonvisitFlag = False
            else:
                pg_rl(device.task, 1)  # update the PG policy for one step
            device.KeyPol.append(device.task.policy)
            tsk0 = copy.deepcopy(device.task)
            device.KeyTsk.append(tsk0)  # tsk0 with the improved policy
            device.KeyReward.append(tsk0.get_value(tsk0.policy['theta']))
            device.KeyAoI.append(tsk0.get_AoI_CPU(tsk0.policy['theta'])[0])
            device.KeyCPU.append(tsk0.get_AoI_CPU(tsk0.policy['theta'])[1])
            device.Keyb.append(tsk0.get_AoI_CPU(tsk0.policy['theta'])[2])
            # 2: Regular (Without Warm Start)
            pg_rl(device.task_Regular, 1)  # update the PG policy for one step
            device.KeyPol_Regular.append(device.task_Regular.policy)
            tsk0_Regular = copy.deepcopy(device.task_Regular)
            device.KeyTsk_Regular.append(tsk0_Regular)
            device.KeyReward_Regular.append(tsk0_Regular.get_value(tsk0_Regular.policy['theta']))
            device.KeyAoI_Regular.append(tsk0_Regular.get_AoI_CPU(tsk0_Regular.policy['theta'])[0])
            device.KeyCPU_Regular.append(tsk0_Regular.get_AoI_CPU(tsk0_Regular.policy['theta'])[1])
            device.Keyb_Regular.append(tsk0_Regular.get_AoI_CPU(tsk0_Regular.policy['theta'])[2])
        else:
            "访问间隔里有1出现"
            for i in np.nonzero(VisitTime)[0]:
                "Iterate所有的1"
                kt = Last_Visted_Time + 1 + i  # kt: key time
                device.KeyTime.append(kt)
                device.ta_dex = device.ta_dex + 1
                if device.ta_dex > self.Devices[action].nTasks - 1:# FXIME: This might come from the shortage of TaskList
                    print("index out!")                      # FIXED: remember to provide a task for the very beginning, i.e. t=0 in Device.gen_TimeTaskList()
                # 1: Warm Start
                device.task = device.TaskList[device.ta_dex]
                device.KeyPol.append(device.task.init_policy)  # For the policy changes not from UAV's update
                tsk0 = copy.deepcopy(device.task)
                device.KeyTsk.append(tsk0)  # tsk0 with initial policy
                device.KeyReward.append(tsk0.get_value(tsk0.init_policy['theta']))
                device.KeyAoI.append(tsk0.get_AoI_CPU(tsk0.init_policy['theta'])[0])
                device.KeyCPU.append(tsk0.get_AoI_CPU(tsk0.init_policy['theta'])[1])
                device.Keyb.append(tsk0.get_AoI_CPU(tsk0.init_policy['theta'])[2])
                # 2: Regular (Without Warm Start)
                device.task_Regular = device.TaskList_Regular[device.ta_dex]
                device.KeyPol_Regular.append(device.task_Regular.init_policy)
                tsk0_Regular = copy.deepcopy(device.task_Regular)
                device.KeyTsk_Regular.append(tsk0_Regular)
                device.KeyReward_Regular.append(tsk0_Regular.get_value(tsk0_Regular.init_policy['theta']))
                device.KeyAoI_Regular.append(tsk0_Regular.get_AoI_CPU(tsk0_Regular.init_policy['theta'])[0])
                device.KeyCPU_Regular.append(tsk0_Regular.get_AoI_CPU(tsk0_Regular.init_policy['theta'])[1])
                device.Keyb_Regular.append(tsk0_Regular.get_AoI_CPU(tsk0_Regular.init_policy['theta'])[2])
                state[action] = t - device.KeyTime[-1]
            if VisitTime[-1] == 0:    # 需要再对t做一个TaDeLL的更新
                device.KeyTime.append(t)
                # 1: Warm Start
                TaDeLL_Model.getDictPolicy_Single(device.task)
                device.KeyPol.append(device.task.policy)
                tsk0 = copy.deepcopy(device.task)
                device.KeyTsk.append(tsk0)  # tsk0 with the improved policy
                device.KeyReward.append(tsk0.get_value(tsk0.policy['theta']))
                device.KeyAoI.append(tsk0.get_AoI_CPU(tsk0.policy['theta'])[0])
                device.KeyCPU.append(tsk0.get_AoI_CPU(tsk0.policy['theta'])[1])
                device.Keyb.append(tsk0.get_AoI_CPU(tsk0.policy['theta'])[2])
                # 2: Regular (Without Warm Start)
                pg_rl(device.task_Regular, 1)  # update the PG policy for one step
                device.KeyPol_Regular.append(device.task_Regular.policy)
                tsk0_Regular = copy.deepcopy(device.task_Regular)
                device.KeyTsk_Regular.append(tsk0_Regular)
                device.KeyReward_Regular.append(tsk0_Regular.get_value(tsk0_Regular.policy['theta']))
                device.KeyAoI_Regular.append(tsk0_Regular.get_AoI_CPU(tsk0_Regular.policy['theta'])[0])
                device.KeyCPU_Regular.append(tsk0_Regular.get_AoI_CPU(tsk0_Regular.policy['theta'])[1])
                device.Keyb_Regular.append(tsk0_Regular.get_AoI_CPU(tsk0_Regular.policy['theta'])[2])



        # FIXME: 不加这个REWARD_REST会不会收敛
        reward_rest = 0
        # for i in range(self.num_Devices):
        #     if i == action:
        #         pass
        #     else:
        #         device = self.Devices[i]
        #         if device.nonvisitFlag:                # if never been visited by UAV
        #             # reward_rest += state[i] * (-5)  # FIXME: choose a proper constant
        #             reward_rest += -5
        #             # 在TIMEUNITS够(100)的情况下，这个数值从 -1 ～ -60 都不会影响reward_, reward_rest的比例, 太大的话就有风险了
        #         else:                             # if this device has been visited by UAV
        #             reward_rest += device.KeyReward[-1]  #  * self.Devices[i].TimeSinceLastVisit # should decrease with time
        #             # reward_rest += state[i] * device.rewards[-1]
        for i in range(self.num_Devices):
            if i != action:
                if device.nonvisitFlag:
                    reward_rest += -10
                else:
                    reward_rest += device.KeyReward[-1]   # should decrease with time
        reward_rest = reward_rest / (self.num_Devices - 1)



        # for index in range(index_start, index_end):
        #     tsk = device.KeyTsk[index]
        #     alpha = device.KeyPol[index]
        #     device.intervals.append(device.KeyTime[index+1] - device.KeyTime[index])
        #     reward = tsk.get_value(
        #         alpha['theta'])  # Didn't use pg_rl() cause it has one step of update which I don't need here
        #     device.rewards.append(reward)
        #     reward_ += device.intervals[-1] * device.rewards[-1]
        device = self.Devices[action]
        index_end = device.KeyTime.index(device.KeyTime[-1])
        # 1: Warm Start rewards history
        if len(device.KeyTime) == 0:
            print('Error captured!')
        reward_ = 0
        AoI_ = 0
        CPU_ = 0
        b_ = 0
        for index in range(index_start, index_end):
            if index + 1 > len(device.KeyTime)-1:
                print('Error captured!')
            interval = device.KeyTime[index + 1] - device.KeyTime[index]
            device.intervals.append(interval)
            if index > len(device.KeyReward)-1:
                print('Error captured!')
            reward_ += device.KeyReward[index] * interval
            """
            FX8: 利用KeyAoI KeyCPU，计算加权均值
            """
            AoI_ += device.KeyAoI[index] * interval
            CPU_ += device.KeyCPU[index] * interval
            b_ += device.Keyb[index] * interval



        """
        FX6:取消这个除的过程，会怎么样呢？还没尝试
        """
        reward_ = reward_ / (device.KeyTime[index_end] - device.KeyTime[index_start]) # not the same as  device.intervals[-1]
        AoI_ = AoI_ / (device.KeyTime[index_end] - device.KeyTime[index_start])
        CPU_ = CPU_ / (device.KeyTime[index_end] - device.KeyTime[index_start])
        b_ = b_ / (device.KeyTime[index_end] - device.KeyTime[index_start])

        # add other devices' reward into account
        """
        FX5: 取消REWARD_REST
        """
        # reward_final = (reward_ + reward_rest)/2  # weighted average,  iteration :discounted reward, back discount
        reward_final = reward_  # weighted average,  iteration :discounted reward, back discount
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

        alpha = param['alpha']
        # reward_fair = 100 * pow(reward_final, 1-alpha) / (1 - alpha)
        # reward_fair = math.log(reward_final)
        """
        FX1
        reward_fair = reward_final + 20
        不需要为了收敛加一个正数，不加也可以照样收敛
        在收敛前，第10个EPISODE的时候，会有一个凸起，不知道与什么有关？
        """
        reward_fair = reward_final

        # FIXME: 不加这个PENALTY这个会收敛吗？
        """
        FX4: 取消访问同一个DEVICE的penalty
        """
        # if action == self.UAV.PositionList[-2]:  # 重复访问的penalty
        #     reward_fair = reward_fair - 200

        mu = param['mu']
        """
        FX2
        这里的reward_fair本来就是要maximize的（在单个TASK里面，我已经翻转过一次了：我要minimize AOI + CPU，相当于maximize和的负）
        所以reward_fair可以直接给到神经网络去maximize
        但是，PV是要minimize的，所以相当于maximize PV的负
        """
        reward_fair1 = (1 - mu) * reward_fair - mu * PV  # 添加飞行能量消耗
        print('reward: ', reward_fair, ', Energy:', PV)

        self.UAV.Reward.append(reward_fair)
        self.UAV.Energy.append(PV)
        self.UAV.Sum_R_E.append(reward_fair1)
        self.UAV.AoI.append(AoI_)
        self.UAV.CPU.append(CPU_)
        self.UAV.b.append(b_)



        # print("done one step")
        # return state, reward_, reward_Regular, t
        return state, reward_, reward_rest, reward_fair1

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
        self.action_head = nn.Linear(64, output_size - 1)
        self.velocity_head = nn.Linear(64, 1)
        # self.velocity_head.weight.requires_grad_(False) # 禁用梯度

        # critic's layer
        # self.value_affine1 = nn.Linear(32, 64)
        self.value_head = nn.Linear(64, 1)

        # action & reward buffer
        self.saved_actions = []
        self.actions = []  # To record the actions
        self.states = []  # To record the states
        self.reward_rewards = []
        self.rewards = []
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
        # x = F.relu(self.affine3(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        # a = F.relu(self.action_affine1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        velocity = torch.sigmoid(self.velocity_head(x))
        print('velocity:', velocity)
        # if velocity < 5:
        #     d = 1
        velocity = max(velocity * self.velocity_lim, velocity / velocity * 5) # 对5的操作，是为了保证左边的velocity依然是TORCH
        # 取下限的另一种方法，可以避免NAN
        # vv[0] = 5
        # velocity = max(velocity * self.velocity_lim, vv) # 对5的操作，是为了保证左边的velocity依然是TORCH
        # 锁定速度值，并确保不出现NAN
        # if velocity.isnan():
        #     velocity[0] = 35
        # velocity[0] = 35  # 用这种方式更保险，避免了出现NAN的情况，强制采用一个固定速度 [10, 15, 20, 25, 30, 35, 40]
        # velocity = velocity / velocity * 35  # 强制采用一个固定速度 [10, 15, 20, 25, 30, 35, 40]
        # critic: evaluates being in the state s_t
        # v = F.relu(self.value_affine1(x))
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values, velocity








