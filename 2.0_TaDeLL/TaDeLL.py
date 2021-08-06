import numpy as np
import math
import spams
from scipy.sparse import csc_matrix
from scipy.linalg import sqrtm
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


    def update(self, task, alpha, hessian):
        """
        This is the function to update TaDeLL model when the optimal policy and corresponding hessian is given.
        :param task:
        :param alpha:
        :param hessian:
        :return:
        """
        """
        Step1: Prepare Hessian
        Step2: Update s
        Step3: Update L and Z
        """
        # Step1: Prepare Hessian
        hh = np.mean(np.abs(hessian))

        return hh




