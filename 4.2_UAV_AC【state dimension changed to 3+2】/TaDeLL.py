import numpy as np
import math
import spams
from scipy.sparse import csc_matrix
from scipy.linalg import sqrtm
from associations import pg_rl


class TaDeLL(object):
    """
    TaDeLL method for one task
    This is the model maintained by central agent, in this case, the UAV.
    Target: update base knowledge L , D, and task specific mapping function s, the latter is maintained by Device
    Given: theta and features
    """

    def __init__(self, d=2, k=2, m=5, mu_one=math.exp(-5), mu_two=math.exp(-5)):
        """
        theta = L @ s  (dx1 = dxk @ kx1)
        phi = D/Z @ s (mx1 = mxk @ kx1)
       :param d: row of L, equals to state space dimension of a single task
       :param k: column of L, the number of latent model components
       :param m: dimension of feature vector for a task
       :param mu_one: in matlab code, 0.001
       :param mu_two: in matlab code, 0.00001  (10 ** (-5))
        """
        self.d = d
        self.k = k
        self.m = m
        self.L = np.random.rand(d, k)
        self.Z = np.random.rand(m, k)

        self.mu_one = mu_one
        self.mu_two = mu_two

        self.T = 0  # Used to count training
        self.counter = 0  # Used to count the update

    def update(self, task, alpha, hessian):
        """
        This is the function to update TaDeLL model when the optimal policy and corresponding hessian is given.
        :param task:
        :param alpha: d x 1
        :param hessian: d x d
        :return:
        """
        """
        Step1: Prepare Hessian
        Step2: Update s
        Step3: Update L and Z
        """
        # Prepare matrix:
        beta = np.concatenate((alpha, task.feature), axis=0)  # (d + m, 1)
        K = np.concatenate((self.L, self.Z), axis=0)  # (d+m, k)

        # The below code is used to justify this function (proved to be right)
        # file_name = 'data.mat'
        # savemat(file_name, {'L': self.L, 'Z': self.Z, 'theta': alpha, 'hessian': hessian, 'feature': task.feature})

        # Step1: Compute s
        h = np.mean(np.abs(hessian))  # (mean number)
        H = h * np.eye(self.m + self.d)  # (d+m, d+m)
        H[0:self.d, 0:self.d] = hessian

        dsqrt = sqrtm(H)  # (d+m, d+m)
        target = dsqrt @ beta  # (d + m, 1)
        dicttransformed = dsqrt @ K  # (d+m, k)

        ss = spams.lasso(np.asfortranarray(target, dtype=np.float64),
                         D=np.asfortranarray(dicttransformed, dtype=np.float64),
                         Q=None, q=None, return_reg_path=False, L=-1, lambda1=self.mu_one / 2.0, lambda2=0.,
                         verbose=False, mode=2)
        s = csc_matrix(ss).toarray()
        task.s = s  # (k, 1)

        # Update L and Z
        L_step = np.mat(alpha @ s.T) * np.mat(s @ s.T + self.mu_two * np.eye(self.k)).I
        L_step = np.array(L_step)
        self.L = self.counter / (self.counter + 1) * self.L + 1/(self.counter+1) * L_step

        Z_step = np.mat(task.feature @ s.T) * np.mat(s @ s.T + self.mu_two * np.eye(self.k)).I
        Z_step = np.array(Z_step)
        self.Z = self.counter / (self.counter + 1) * self.Z + 1/(self.counter+1) * Z_step

        self.counter = self.counter + 1

        return self.L, self.Z, task.s

    def train(self, tasks):
        """
        Iterate unitl all tasks are visited.
        Each iteration, learn one step of PG and update L, Z, s, using task's alpha and hessian, and feature.
        :param tasks:
        :return:
        """
        print("training process starting:")
        # tasks_pre = copy.deepcopy(tasks)  # Prepare tasks
        observe_flag = 0
        observed_tasks = np.zeros(np.shape(tasks)[0])
        counter = 1
        LimOne = 0
        LimTwo = np.shape(tasks)[0]

        while not observe_flag:
            # choose a task
            if np.all(observed_tasks):
                observe_flag = 1
                # print('All tasks have been observed')

            taskID = np.random.randint(LimOne, LimTwo)
            task = tasks[taskID]

            if observed_tasks[taskID] == 0:
                observed_tasks[taskID] = 1
                self.T = self.T + 1

            pg_rl(task, 1)  # Execute one step of PG

            # prepare the task hessian and alpha
            hessian = task.get_hessian(task.policy["theta"])
            task.hessian_matrix = hessian

            # Update TaDeLL Model
            self.update(task, task.policy["theta"], task.hessian_matrix)

            counter = counter + 1
            print("training Count @", counter)

        print("Training process finishes")

    def comp_mu_sig(self, tasks):
        """
        Compute mu and sig (variance) using a set of tasks (could be all tasks, training tasks, testing tasks)
        :param tasks: the set of tasks to compute mu, sig
        :return:
        """

        plain_feature_mat = np.hstack((tasks[0].plain_feature, tasks[1].plain_feature))  # stack horizontally
        for i in range(2, len(tasks)):
            plain_feature_mat = np.hstack((plain_feature_mat, tasks[i].plain_feature))  # (m,num_of_tasks)

        mu = np.mean(plain_feature_mat, 1)  # (m,)
        sig = np.var(plain_feature_mat, 1, ddof=1)  # (m,)

        mu = np.reshape(mu, (self.m, 1))  # (m,1)
        sig = np.reshape(sig, (self.m, 1))  # (m,1)

        return mu, sig

    def test(self, tasks):
        """
        Given testing tasks, execute PG using warm_start_policy
        :param tasks:
        :return:
        """
        pass

    def getDictPolicy(self, tasks):
        """
        Given a task, obtain the warm start policy using task's features, model's L and Z.
        :param tasks:
        :return:
        """
        L = np.mat(self.L)
        Z = np.mat(self.Z)
        for i in range(0, len(tasks)):
            task = tasks[i]
            feature = np.mat(task.feature)
            theta_hat = L * (Z.I * feature)
            task.policy['theta'] = np.array(theta_hat)

        print("get dict policy")

    def getDictPolicy_Single(self, task):
        L = np.mat(self.L)
        Z = np.mat(self.Z)
        feature = np.mat(task.feature)
        theta_hat = L * (Z.I * feature)
        task.policy['theta'] = np.array(theta_hat)
