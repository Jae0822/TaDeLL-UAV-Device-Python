import numpy as np
import math
import spams
from scipy.sparse import csc_matrix
from scipy.linalg import sqrtm
from scipy.io import savemat
# import sklearn
# from sklearn.linear_model import LassoLars
# from scipy.stats import truncnorm
# from scipy import linalg


class TaDeLL(object):
    """
    TaDeLL method for one task
    This is the model maintained by central agent, in this case, the UAV.
    Target: update base knowledge L , D, and task specific mapping function s, the latter is maintained by Device
    Given: theta and features
    """

    def __init__(self, d=2, k=2, m=5, mu_one=math.exp(-5), mu_two=math.exp(-5)):
        """
        \theta = L @ s  (dx1 = dxk @ kx1)
        \phi = D/Z @ s (mx1 = mxk @ kx1)
       :param d: row of L, equals to state space dimension of a single task
       :param k: column of L, the number of latent model components
       :param m: dimension of feature vector for a task
       :param mu_one: in matlab code, 0.001
       :param mu_two: in matlab code, 0.00001
        """
        self.d = d
        self.k = k
        self.m = m
        self.L = np.random.rand(d,k)
        self.Z = np.random.rand(m,k)

        self.mu_one = mu_one
        self.mu_two = mu_two

        self.T = 0
        self.counter = 0


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

        file_name = 'data.mat'
        savemat(file_name, {'L': self.L, 'Z': self.Z, 'theta': alpha, 'hessian': hessian, 'feature': task.feature})

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

        print("hello")

        return self.L, self.Z, task.s


    def train(self, Tasks):
        pass;


