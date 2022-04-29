import numpy as np
import copy
from scipy.stats import truncnorm
import math
import random
from env import Task
from associations import pg_rl


class Device(object):
    def __init__(self, mu, sig, nTimeUnits=900, nTasks=3, edge=1000):
        """
        Basic attributes of a Device
        :param nTasks: Number of tasks for one device in the studied time period
        :param nTimeUnits:
        :param edge: the length of the square, unit: m
        :param mu: used to normalize the plain features
        :param sig: used to normalize the plain features
        """
        self.nTimeUnits = nTimeUnits
        self.nTasks = nTasks
        self.edge = edge
        self.location = edge * np.random.random_sample((2, 1))

        # Task list, Time List, current attributes
        self.TimeList = self.gen_TimeList(nTimeUnits, nTasks)  # The list of time that indicates the arrival of a new task
        self.TaskList = self.gen_TaskList(nTasks, mu, sig)
        self.NewTaskArrival = np.where(self.TimeList)[0]  # The list of New task arrival time
        self.ta_dex = 0  # current task index
        self.task = self.TaskList[self.ta_dex]  # current task
        self.TaskList_Natural = copy.deepcopy(self.TaskList)  # For the comparison without warm start
        self.task_Natural = self.TaskList_Natural[self.ta_dex]  # current task

        # The record history (generated during running)
        self.KeyTime = [0]  # The list of key time at which the policy changes (1. UAV visits 2. new task arrival)
        self.KeyPol = [self.TaskList[0].init_policy]  # The list of policy at/after key time slot
        tsk0 = copy.deepcopy(self.TaskList[0])
        self.KeyTsk = [tsk0]
        self.flag = True  # To indicate the first visit
        self.rewards = []
        self.rewards_Natural = []
        self.KeyPol_Natural = copy.deepcopy(self.KeyPol)  # The Key points for natural learning without warm start
        self.KeyTsk_Natural = copy.deepcopy(self.KeyTsk)

    def gen_TimeList(self, nTimeUnits, nTasks):
        """
        To make sure the time units are not too close
        Also make sure that they are normally distributed
        :param nTimeUnits:
        :param nTasks:
        :return:
        """
        """
        FIXME: There exits a possibility that the number of "1" might less than "nTasks-1"
        """
        TimeList = np.zeros(nTimeUnits)
        # TimeList[0] = 1
        mean = nTimeUnits / nTasks
        t = int(np.random.normal(mean, mean / 10))
        # while t < mean/5:
        #     t = int(np.random.normal(mean, mean/5))
        for i in range(nTasks-1):
            TimeList[t] = 1
            t = t + int(np.random.normal(mean, mean / 10))
            # while t < mean/5:
            #     t = t + int(np.random.normal(mean, mean/5))
        # Remove the last/extra one that's too close to the end
        if len(np.where(TimeList)[0]) > nTasks - 1:
            TimeList[np.where(TimeList)[0][-1]] = 0  # Seems never runs to here
        return TimeList

    def gen_TaskList(self, nTasks, mu, sig):
        TaskList = []  # List of tasks
        for i in range(nTasks):
            task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
                        cpu_max=random.randint(30, 70), p=0.4 * np.random.random_sample() + 0.3, d=2, k=2)
            task.extract_feature(mu, sig)  # Normalize the plain_feature using mu, sig. self.feature will be updated.
            TaskList.append(task)
        return TaskList



class UAV_Obj(object):
    def __init__(self, V):
        self.init_location = np.zeros((2, 1))
        self.location = np.zeros((2, 1))
        self.V = V  # m/s

    def nextpoint(self, Devices, profit_list, PolType):
        de_dex = None
        NPoint = None
        if PolType == "Random":
            de_dex = np.random.randint(len(Devices))
            NPoint = Devices[de_dex].location
            while np.linalg.norm(self.location - NPoint) < .1:  # if next point is the same as the current point
                de_dex = np.random.randint(len(Devices))
                NPoint = Devices[de_dex].location
        else:
            pass
        return NPoint, de_dex

    def compRewards(self, Devices):
        """
        To compute/expand the rewards over time using KeyTime and KeyPol of each device
        :param Devices:
        :return:
        """
        # 1: Warm Start rewards history
        Rewards = []
        for device in Devices:
            t = 0
            index = 0
            for t in range(device.nTimeUnits):
                if index != (len(device.KeyTime) - 1):  # To avoid the last index problem
                    if t >= device.KeyTime[index + 1]:
                        index = index + 1
                tsk = device.KeyTsk[index]
                alpha = device.KeyPol[index]
                reward = tsk.get_value(alpha['theta'])  # Cause the pg_rl has one step of update which I don't need here
                device.rewards.append(reward)
            Rewards.append(device.rewards)

        # 2: Natural (Without Warm Start)
        Rewards_Natural = []
        for device in Devices:
            t = 0
            index = 0
            for t in range(device.nTimeUnits):
                if index != (len(device.KeyTime) - 1):  # To avoid the last index problem
                    if t >= device.KeyTime[index + 1]:
                        index = index + 1
                tsk = device.KeyTsk_Natural[index]
                alpha = device.KeyPol_Natural[index]
                reward = tsk.get_value(alpha['theta'])  # Cause the pg_rl has one step of update which I don't need here
                device.rewards_Natural.append(reward)
            Rewards_Natural.append(device.rewards_Natural)

        return Rewards, Rewards_Natural

    def UAV_Fly(self, Devices, TaDeLL_Model, PolType, **param):
        # Unpacking the parameter dictionary
        nTimeUnits = param['nTimeUnits']
        nTasks = param['nTasks']

        # Preparing relevant parameters

        profit_list = []
        CPoint = self.init_location

        t = 0

        # Decide the next point and fly
        NPoint, de_dex = self.nextpoint(Devices, profit_list, PolType)  # coordinate, device index
        distance = np.linalg.norm(CPoint - NPoint)  # Compute the distance of two points
        Fly_time = round(distance / self.V)
        CPoint = NPoint  # UAV flies to next point
        self.location = NPoint  # UAV flies to the destination

        device = Devices[de_dex]

        t = t + Fly_time

        while t < nTimeUnits:

            # Update policy for the current device
            if device.flag and (not np.any(device.TimeList[device.KeyTime[-1]: t])):
                # it's first time and there's no more new task arrival before this visit (still the first/initial task)
                device.flag = False
                device.KeyTime.append(t)

                # 1: Warm Start
                TaDeLL_Model.getDictPolicy_Single(device.task)
                device.KeyPol.append(device.task.policy)
                tsk0 = copy.deepcopy(device.task)
                device.KeyTsk.append(tsk0)

                # 2: Natural (Without Warm Start)
                pg_rl(device.task_Natural, 1)  # update the PG policy for one step
                device.KeyPol_Natural.append(device.task_Natural.policy)
                tsk0_Natural = copy.deepcopy(device.task_Natural)
                device.KeyTsk_Natural.append(tsk0_Natural)

            elif np.any(device.TimeList[device.KeyTime[-1]: t]):  # when there's a new task arrived
                if device.flag:  # If it's the first time visit. But there's a second new task arrives
                    device.flag = False
                device.ta_dex = device.ta_dex + 1
                if device.ta_dex > nTasks - 1:
                    print("index out!")  # FXIME: to debug an error. But little chance to encounter the bug
                    # FIXED: The second elif shouldn't exist.
                device.task = device.TaskList[device.ta_dex]
                device.task_Natural = device.TaskList_Natural[device.ta_dex]

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

                # 2: Natural (Without Warm Start)
                device.KeyPol_Natural.append(device.task_Natural.init_policy)
                tsk0_Natural = copy.deepcopy(device.task_Natural)
                device.KeyTsk_Natural.append(tsk0_Natural)
                pg_rl(device.task_Natural, 1)
                device.KeyPol_Natural.append(device.task_Natural.policy)
                tsk0_Natural = copy.deepcopy(device.task_Natural)
                device.KeyTsk_Natural.append(tsk0_Natural)


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
                # device.task = device.TaskList[device.ta_dex]
                device.KeyTime.append(t)

                # 1: Warm Start
                pg_rl(device.task, 1)  # update the PG policy for one step
                device.KeyPol.append(device.task.policy)
                tsk0 = copy.deepcopy(device.task)
                device.KeyTsk.append(tsk0)  # tsk0 with the improved policy

                # 2: Natural (Warm Start)
                pg_rl(device.task_Natural, 1)
                device.KeyPol_Natural.append(device.task_Natural.policy)
                tsk0_Natural = copy.deepcopy(device.task_Natural)
                device.KeyTsk_Natural.append(tsk0_Natural)


            NPoint, de_dex = self.nextpoint(Devices, profit_list, PolType)  # coordinate, device index
            distance = np.linalg.norm(CPoint - NPoint)  # Compute the distance of two points
            Fly_time = round(distance / self.V)
            CPoint = NPoint  # UAV flies to next point
            self.location = NPoint  # UAV flies to the destination

            device = Devices[de_dex]

            t = t + Fly_time

        return Devices