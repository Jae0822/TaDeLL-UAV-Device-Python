import numpy as np
import copy
from scipy.stats import truncnorm
import math
# from scipy import linalg


class Task(object):
    """
    Device is able to obtain data trajectories
    Inputs contain two sorts of variables: device related and environment related
    """

    def __init__(self, mean_packet_cycles=20, variance_packet_cycles=4, cpu_max=50, p=0.5, d=2, k=2, alpha_CPU=0.001):
        """
        ArrivalRate: lambda_i for device i, Poisson distribution
        MeanCycles:  averaged data packets cycles
        variance_packet_cycles: variance of the size for data packets
        CPUMax: Maximum CPU cycles for one time slot
        p: The probability of Bernoulli distribution over each time slot (Either Poisson or Bernoulli)
        """
        # A label of visit/learn by UAV/Agent
        self.label = False

        # Environment related:
        # self.ArrivalRate = ArrivalRate
        self.mean_packet_cycles = mean_packet_cycles  # self.mean_packet_cycles = np.random.uniform(30,100)
        # Averaged CPU cycles required for each data packet
        self.variance_packet_cycles = variance_packet_cycles  # standard deviation of self.PacketCycles
        self.cpu_max = cpu_max
        self.p = p  # np.random.rand() # The probability of Bernoulli distribution over each time slot

        # Device related:
        self.d = d  # dimension of L, and state and policy dimension.
        # Please check here:https://sm.ms/image/QCotEY8lJwr9Z2h
        self.k = k  # dimension of L, number of elements.  See above link
        self.alpha_CPU = alpha_CPU  # 0.0001  # chip architecture

        self.pending_packets = {'queue': [],
                                'time_stamp': []}
        # 1. The queue of pending data packets at the device, each element is the CPU cycles required.
        # 2. The queue of the time stamp of the pending data packets
        self.t = 0  # The number of time slot

        # Trajectory and target related:
        self.rollouts = 100  # Number of rollouts
        self.trajectory = 50  # Number of steps in a rollout
        self.beta = 0.5  # AoI and energy consumption trade-off

        # Policy-Gradient initialization:
        self.learning_rate = 5.0
        self.gamma = 0.9  # Fading/Decaying coefficient of the reward
        # FIXME: two different ways to initialize initial policy for policy gradient method
        # (1）zeros: as in matlab paper
        # (2) np.random.rand(): uniform distribution as in matlab
        # (3) np.random.randn(): standard normal distribution as in ELLA.py by Paul Ruvolo
        # FIXME: sigma: np.random.normal(0,sigma**2), or uniform distribution between (0,1)
        # self.init_policy = {'theta': np.array([[0.77027409], [0.94257635]]), 'sigma': 0.9719}
        # self.init_policy = {'theta': np.array([[0.75930339], [0.78251577]]), 'sigma': 5}
        # self.init_policy = {'theta': np.array([[0.1030141 ], [0.54868897]]), 'sigma': 5}
        # self.init_policy = {'theta': np.array([[0.4296686], [0.02653781]]), 'sigma': 5}
        # FIXME: sigma is too large
        self.init_policy = {'theta': np.random.rand(self.d, 1), 'sigma': 5}  # mapping function of the device
        self.policy = copy.deepcopy(self.init_policy)  # the policy which is going to be learned by PGELLA
        # init_policy and policy has their 'theta' with k arrays, each array has one element

        # PG-ELLA related:
        self.hessian_matrix = None  # (self.d, self.d)
        # self.alpha_matrix = np.zeros((self.d, 1))  # (self.d, 1)
        self.s = np.zeros((self.k, 1))  # (self.k, 1)

    def reset(self):
        self.t = 0
        self.pending_packets = {'queue': [], 'time_stamp': []}

    def draw_init_state(self):
        aoi = np.zeros(1)
        # The arrival of a data packet follows Bernoulli distribution
        # d = 1 indicates a data packet arrival, d = 0, otherwise.
        d = np.random.binomial(1, self.p, 1)
        # b is drawn from a truncated normal distribution: N~(self.mean_packet_cycles, self.variance_packet_cycles)
        b = np.zeros(1)
        # if self.d > 0:
        #     while b < 0:
        #         # b = np.random.normal(self.mean_packet_cycles, self.variance_packet_cycles)
        #         b = self.mean_packet_cycles
        # else:
        #     low, up = (0 - self.mean_packet_cycles) / self.variance_packet_cycles, (
        #             np.inf - self.mean_packet_cycles) / self.variance_packet_cycles
        #     b = self.mean_packet_cycles + np.sqrt(self.variance_packet_cycles) * truncnorm.rvs(low, up, loc=0, scale=1,
        #                                                                                    size=1, random_state=None)
        # b = d * np.around(np.random.normal(self.mean_packet_cycles, self.packet_variance, size = 1))
        if d == 1:
            b = d * np.random.normal(self.mean_packet_cycles, self.variance_packet_cycles)
            # b = d * self.mean_packet_cycles
            # b = d * b
            self.pending_packets['queue'].append(b[0])
            self.pending_packets['time_stamp'].append(0)
        # state is an array of one dimension, i.e. vector
        state = np.concatenate((aoi, b))
        return state  # (2,)

    def draw_action(self, state, alpha):
        """
        Gaussian Policy:
        action = policy['theta'] * state + policy['sigma'],
        with low and up value truncated between [0 + eps, self.cpu_max]
        """
        state = np.reshape(state, (1, 2))
        # 1x2 @ 2x1 = 1x1.
        my_mean = np.dot(state, alpha)
        if math.isnan(my_mean[0]):
            print("draw action nan")
        # my_mean = np.array([state]) @ self.policy['theta'] # 1x2 @ 2x1 = 1x1
        my_std = self.policy['sigma']
        # if my_mean < 0:
        #     raise ValueError("negative action")
        if self.d > 0:
            # generate normal distribution, cut two ends
            action = np.random.normal(my_mean[0][0], my_std)
            clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
            # FIXME: will min value 0 affect the computing? Or should I use a minimal eps?
            action = clamp(action, 0, self.cpu_max)
            action = np.array([action])
        else:
            # as in matlab code
            a, b = (0 - my_mean) / my_std, (self.cpu_max - my_mean) / my_std
            # truncnorm can generate a truncated normal continuous random variable.
            action = my_mean + np.sqrt(my_std) * truncnorm.rvs(a, b, loc=0, scale=1, size=1, random_state=None)
        return action  # 0x1 nparray:(1,)

    def step(self, state, _action):
        self.t = self.t + 1
        aoi, b = state
        aoi = aoi + 1
        action = _action[0]  # Extract the action from an array to s single number

        # Two different ways to process pending data packets in a row
        if self.d > 0:
            while len(self.pending_packets['queue']) > 0:
                if action >= self.pending_packets['queue'][0]:
                    action = action - self.pending_packets['queue'][0]
                    aoi = self.t - self.pending_packets['time_stamp'][0]  # The temporary AoI
                    del self.pending_packets['queue'][0]  # The first packet is already finished
                    del self.pending_packets['time_stamp'][0]
                else:
                    self.pending_packets['queue'][0] = self.pending_packets['queue'][0] - action
                    # aoi = aoi + 1 shouldn't be here. Cause aoi = t - time_stamp already makes AoI correct in some case.
                    break
        else:
            pass
            # for index, size in enumerate(self.pending_packets['queue']):
            #     if action < size:
            #         self.pending_packets['queue'][0] = size - action
            #         break
            #     else:
            #         if len(self.pending_packets['queue']) > 1:  # when another data packet is pending
            #             #  self.pending_packets['queue'][1] = self.pending_packets['queue'][1] - (action - size)
            #             action = action - size  # Update the remaining computing capability to the next packet
            #             # AoI is updated because of the processing for a new data packet is finished.
            #             aoi = self.t - self.pending_packets['time_stamp'][0]
            #             # FIXME: The remained CPU can not be transferred to next packet.
            #             # Because the first element of the list is removed, which makes the list to be empty (no more for loop)
            #             # Or xxxx. The for loop is not right here anyway.
            #             del self.pending_packets['queue'][0]  # The first packet is already finished
            #             del self.pending_packets['time_stamp'][0]
            #         else:
            #             # FIXME:  Is this "else" really necessary?
            #             # FIXME: If only one packet is left and its processing is finished.
            #             # AoI is updated because of the processing for a new data packet is finished.
            #             aoi = self.t - self.pending_packets['time_stamp'][0]
            #             del self.pending_packets['queue'][0]  # The first packet is already finished
            #             del self.pending_packets['time_stamp'][0]
            #             break

        # FIXME: This implies the pending data packet is also a part of the state
        # The arrival of a new data packet
        d = np.random.binomial(1, self.p, 1)
        new_b = 0
        # if self.d > 0:
        #     while new_b < 0:
        #         # new_b = np.random.normal(self.mean_packet_cycles, self.variance_packet_cycles)
        #         new_b = self.mean_packet_cycles
        # else:
        #     # Why there's a low, up here? Prevent negative value of new_b
        #     low, up = (0 - self.mean_packet_cycles) / self.variance_packet_cycles, (
        #             np.inf - self.mean_packet_cycles) / self.variance_packet_cycles
        #     new_b = self.mean_packet_cycles + np.sqrt(self.variance_packet_cycles) * truncnorm.rvs(low, up, loc=0,
        #                                                                                            scale=1,
        #                                                                                            size=1,
        #                                                                                            random_state=None)
        # new_b = d * new_b
        # Add the new arrived data packet to the pending queue
        if d == 1:
            new_b = d * np.random.normal(self.mean_packet_cycles, self.variance_packet_cycles)
            # new_b = d * self.mean_packet_cycles
            self.pending_packets['queue'].append(new_b[0])
            self.pending_packets['time_stamp'].append(self.t)

        # Summing all the pending packets
        b = sum(self.pending_packets['queue'])  # The remaining cycles is the sum of remained packets
        # next_state = np.concatenate((aoi, b))
        next_state = np.array([aoi, b])
        reward = - (self.beta * aoi + (1 - self.beta) * self.alpha_CPU * _action[0] ** 3)
        return next_state, reward  # 0x2, number(float)

    def collect_path(self, alpha):
        path = []

        for i in range(self.rollouts):
            self.reset()
            states = []
            actions = []
            rewards = []
            state = self.draw_init_state()
            states.append(state)

            for j in range(self.trajectory):
                action = self.draw_action(state, alpha)
                next_state, reward = self.step(state, action)
                actions.append(action)
                rewards.append(reward)
                states.append(next_state)
                state = next_state
            path.append(dict(
                states=np.array(states),
                actions=np.array(actions),
                rewards=np.array(rewards)
            ))
        return path

    def get_value(self, alpha):
        # get the mean of a trajectory over all trajectories. Rather than adding all steps together
        # mean(all trajectory) expected(all steps) reward(trajectory, step)
        path = self.collect_path(alpha)
        value = 0
        for i in range(len(path)):
            value = value + np.mean(path[i]["rewards"])
        # value = value / len(path)
        return value

    def djd_nac(self, path):
        # """
        # Programmed by gabrieledcjr: https://github.com/IRLL/Vertical_ARDrone/blob/63a6cc455cb8092f6ab88cc68dbd71feff361b51/src/vertical_control/scripts/pg_ella/episodicNaturalActorCritic.py
        # theta  = \theta + \alpha_CPU \Nabla J(\theta)
        # This function gives \Nabla J(\theta) as the return based on NAC(Natural Actor Critic) method.
        # After calling this function, a gradient step (here is the policy step) can be obtained.
        # input: path
        # output: the policy step: w
        # """
        # FIXME: The djd_nac here is computed based on each reward in a trajectory, rather than the summation of
        #  all rewards in a single trajectory (as in matlab code and in the PG-ELLA paper)
        # FIXME: >>>But apparently this can work too. <<<
        # FIXME: value of gamma should be reconsidered.
        # Obtain expected return
        j = 0
        if self.gamma == 1:  # This is gonna be skipped cause gamma = 0.9
            for trail in range(len(path)):
                j = j + np.mean(path[trail]['rewards'])
            j = j / len(path)
        # Obtain gradients
        mat = np.zeros((len(path), self.d + 1))
        vec = np.zeros((len(path), 1))
        theta = self.policy['theta']
        for trail in range(len(path)):
            state = np.reshape(path[trail]['states'][0], (1, 2))
            action = path[trail]['actions'][0]
            der = ((action - state @ theta) @ state) / (self.policy['sigma'] ** 2)  # (2,)
            mat[trail] = np.append(np.zeros(np.shape(der)), np.array([1]))
            vec[trail][0] = 0
            for step in range(len(path[trail]['rewards'])):
                state = np.reshape(path[trail]['states'][step], (1, 2))  # (1, 2)
                action = path[trail]['actions'][step]  # (1, 1)
                # theta = self.policy['theta'] # (2,1)
                der = ((action - state @ theta) @ state) / (self.policy['sigma'] ** 2)   # (2,)
                mat[trail] = mat[trail] + np.append(der, np.array([0]))
                if math.isnan(mat[trail][0]) or math.isnan(mat[trail][1]):
                    print("nan found in mat")
                vec[trail] = vec[trail] + (path[trail]['rewards'][step])
        # cond(Mat)
        nrm = np.diag(np.append(1 / np.std(mat[:, 0:self.d], ddof=1, axis=0), np.array([1])))  # (3,3) or (self.d + 1, self.d + 1)
        w0 = nrm @ np.linalg.inv(nrm @ mat.T @ mat @ nrm) @ nrm @ mat.T @ vec  # (3,1) or (self.d + 1, 1)
        w = w0[0:(np.max(np.shape(w0)) - 1)]  # (2,1) or (self.d , 1) The same size as policy['theta']
        if math.isnan(w[0]) or math.isnan(w[1]):
            print("nan found in w")
        return w

    # def djd_reinforce(self, path):
    #     """
    #     \theta  = \theta + \alpha_CPU \Nabla J(\theta)
    #     This function gives \Nabla J(\theta) as the return based on REINFORCE method.
    #     After calling this function, a gradient step (here is the policy step) can be obtained.
    #     """
    #     pass

    def gaussian_grad(self, state, action):
        """
        Programmed by cscsai: https://github.com/cdcsai/Online_Multi_Task_Learning
        Compute the gradient of the gaussian policy.
        sigma^(-2) * (action - theta.T * state) * state

        input: state 0x2, action: 0x1, policy['theta']: 2x1
        output: gradient of gaussian policy: 2x1
        """
        state = np.reshape(state, (2, 1))
        return ((action - self.policy['theta'].T @ state) @ state.T).T  # 2x1 (the same as policy['theta'])

    def get_alpha(self):
        pass

    def get_hessian(self, alpha):
        """
        Get Hessian Matrix of Gaussian Policy
        Follows the formula in paper PG-ELLA. There's one typo in Section 5.1: a reward of one trajectory is also required in the hessian matrix.
        There are two ways of computing. One is based on one state by state.
        The other is to compute all by once as in matlab code: computeHessiaDeviceM and python code: https://github.com/IRLL/Vertical_ARDrone/blob/63a6cc455cb8092f6ab88cc68dbd71feff361b51/src/vertical_control/scripts/pg_ella/computeHessian.py
        """
        hes = np.zeros((self.d, self.d))
        path = self.collect_path(alpha)
        for trail in range(len(path)):
            adv = 0
            rew = np.sum(path[trail]["rewards"])  # float
            for step in range(len(path[trail]["rewards"])):
                state0 = path[trail]["states"][step]  # (self.d,)
                state = np.reshape(state0, (2, 1))  # (self.d, 1)
                adv = adv + state @ state.T  # (2,2) (self.d, self.d)
            mat = rew * adv / (self.policy["sigma"] ** 2)
            hes = hes + mat
        hes = - hes / len(path)
        # self.hessian_matrix = hes  # (2,2) (self.d, self.d)
        return hes


class Device(object):
    def __init__(self):
        self.location = np.zeros(2,1)
        self.task = Task(mean_packet_cycles=10, variance_packet_cycles=3, cpu_max=50, p=0.5, d=2, k=2)