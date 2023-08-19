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
        TimeList = np.zeros(nTimeUnits)
        TaskList = []
        TaskList.append(self.taskList_set[1])
        mean = self.frequency
        t = int(np.random.normal(mean, mean / 10))
        while t < nTimeUnits:
            TimeList[t] = 1
            TaskList.append(self.taskList_set[1])
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

        with open('input_files/TaDeLL_result_k_2.pkl', 'rb') as f:
            _, _, _, self.TaDeLL_Model, _, _, _, _, _ = pickle.load(f)

        self.initialization(Devices, UAV)


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
        index_start = self.update_device_since_keytime(cur_device, t)

        # update reward
        reward_rest = 0
        for device in self.Devices: 
            if device != cur_device:
                if device.nonvisitFlag:
                    reward_rest += -10
                else:
                    reward_rest += device.KeyReward[-1]   # should decrease with time
        print("reward_rest: {}".format(reward_rest))
        reward_rest = reward_rest / max(1, (self.num_Devices - 1)) #avoid dividing by 0 if only 1 device


        index_end = cur_device.KeyTime.index(cur_device.KeyTime[-1])

        reward_ = 0
        AoI_ = 0
        CPU_ = 0
        b_ = 0
        for index in range(index_start, index_end):
            if index + 1 > len(cur_device.KeyTime)-1:
                print('Error captured!')
            interval = cur_device.KeyTime[index + 1] - cur_device.KeyTime[index]
            cur_device.intervals.append(interval)
            if index > len(cur_device.KeyReward)-1:
                print('Error captured!')
            print("updating reward: {} inteval: {}".format(cur_device.KeyReward[index], interval))
            reward_ += cur_device.KeyReward[index] * interval
            """
            FX8: 利用KeyAoI KeyCPU，计算加权均值
            """
            AoI_ += cur_device.KeyAoI[index] * interval
            CPU_ += cur_device.KeyCPU[index] * interval
            b_ += cur_device.Keyb[index] * interval



        """
        FX6:取消这个除的过程，会怎么样呢？还没尝试
        """
        reward_ = reward_ / (cur_device.KeyTime[index_end] - cur_device.KeyTime[index_start]) # not the same as  device.intervals[-1]
        AoI_ = AoI_ / (cur_device.KeyTime[index_end] - cur_device.KeyTime[index_start])
        CPU_ = CPU_ / (cur_device.KeyTime[index_end] - cur_device.KeyTime[index_start])
        b_ = b_ / (cur_device.KeyTime[index_end] - cur_device.KeyTime[index_start])

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

    def update_device_since_keytime(self, device, t):
        device.nonvisitFlag = False
        last_visited_time = device.KeyTime[-1]
        index_start = device.KeyTime.index(last_visited_time)

        VisitTime = device.TimeList[last_visited_time+1 : t+1]

        # ------------------Update estimated reward for rest devices----------------------#
        # FIXME: Fly_Time or state[i] （两种模式哪一种更好）
        # FIXME: end of the nTimeUnits, 怎么处理剩下的一小部分或多或少的时间。

        if not np.any(VisitTime):
            "两次访问之间为全0，没有任何新任务出现"
            device.KeyTime.append(t)
            # 1: Warm Start
            if device.task.nonvisitFlag:
                self.TaDeLL_Model.getDictPolicy_Single(device.task)
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
                kt = last_visited_time + 1 + i  # kt: key time
                device.KeyTime.append(kt)
                device.ta_dex = device.ta_dex + 1
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
            if VisitTime[-1] == 0:    # 需要再对t做一个TaDeLL的更新
                device.KeyTime.append(t)
                # 1: Warm Start
                self.TaDeLL_Model.getDictPolicy_Single(device.task)
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
        return index_start


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
        state_values = self.value_head(x)
        velocity = torch.clamp(self.velocity_head(x), 5, self.velocity_lim)
        print('velocity:', velocity)
        # 锁定速度值，并确保不出现NAN
        if velocity.isnan() or velocity.isinf():
            luck = 1
            # velocity[0] = 25

        if torch.isnan(action_prob).any() or torch.isinf(action_prob).any():
            luck = 1
        if torch.isnan(state_values).any() or torch.isinf(state_values).any():
            luck = 1
        if torch.isnan(velocity).any() or torch.isinf(velocity).any():
            luck = 1

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values, velocity
