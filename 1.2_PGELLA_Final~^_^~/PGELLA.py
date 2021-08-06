import numpy as np
import math
import spams
from scipy.sparse import csc_matrix
from scipy.linalg import sqrtm
# import sklearn
# from sklearn.linear_model import LassoLars
# from scipy.stats import truncnorm
# from scipy import linalg

"""
Important Tips!
All the FIXME in this script are fixed. Don't worry!
"""


class Model(object):
    """
    Online PG-ELLA model update function for a single task.
    This is the model maintained by central agent, in this case, the UAV.
    Target: update base knowledge L and task specific mapping function s, the latter is maintained by Device
    """

    def __init__(self, d=2, k=2, mu_one=math.exp(-5), mu_two=math.exp(-5)):
        """
        d: row of L, equals to state space dimension of a single task
        k: column of L, the number of latent model components
        """
        self.d = d
        self.k = k
        # FIXME: Three potential ways to initialize L
        # FIXED: The (2) works fine. Stick to it.
        # (1) zeros: as in paper PG-ELLA and ELLA
        # (2) rand: [0,1),Average (uniform) distribution. as in cross-domain matlab code by Bou Ammar, also adopted by zhenz --- np.random.rand(self.d, self.k)
        # (3) randn: standard normal (Gaussian) distribution, as in ELLA.py by Paul Ruvolo --- np.random.randn(d, k)
        self.L = np.random.rand(self.d, self.k)
        self.A = np.zeros((d * k, d * k))
        self.b = np.zeros((d * k, 1))

        self.mu_one = mu_one
        self.mu_two = mu_two

        self.T = 0  # The number of tasks encountered so far.

    def update_model(self, task, alpha, hessian):
        """
        Takes a single device as the input for model update.
        The base knowledge L and mapping function of the device will be updated.
        """
        """
        There are two ways to update the model:
        A. As in matlab code and the online PG-ELLA paper. 
        B. As in python code by @author: brownyoda: https://github.com/IRLL/Vertical_ARDrone/blob/63a6cc455cb8092f6ab88cc68dbd71feff361b51/src/vertical_control/scripts/pg_ella/updatePGELLA.py
        """
        # alpha = task.policy["theta"]  # (self.d, 1)
        # hessian = task.hessian_matrix  # (self.d, self.d)
        s = task.s  # (self.k, 1)

        # print("--------------------------")
        # print("the original s:", s)
        # print("the original A:")
        # print(self.A)
        # print("the original b:")
        # print(self.b)
        # print("the original L:")
        # print(self.L)

        # is this the first time for the agent to encounter the task
        if task.label:  # if this task is encountered before
            self.A = self.A - np.kron(s @ s.T, hessian)  # (k*d, k*d)
            self.b = self.b - (np.kron(s.T, alpha.T @ hessian)).T  # (k*d, 1)
        else:
            task.label = True  # If this is a new task, set the task label to True.
            self.T = self.T + 1  # Add this new task to task numbers encountered so far.

        # print("the revised A:")
        # print(self.A)
        # print("the revised b:")
        # print(self.b)

        # Reinitialize "All Zero" Columns for L
        # FIXME: The reinitialization of L should be the same as the definition of L in init function
        # FIXED: Yes, You did it.
        for i in np.where(~self.L.any(axis=0))[0]:  # Find all zero columns
            # print("the original L: ", self.L)
            self.L[:, i] = np.random.rand(self.d)
        # print("the reinitialized L:")
        # print(self.L)

        # Update s
        dsqrt = sqrtm(hessian)  # (self.d, self.d)
        target = dsqrt @ alpha  # (d, 1)
        dicttransformed = dsqrt @ self.L  # (d, k)
        # lasso(X, D=None, Q=None, q=None, return_reg_path=False, L=-1, lambda1=None, lambda2=0.,
        #       mode=spams_wrap.PENALTY, pos=False, ols=False, numThreads=-1,
        #       max_length_path=-1, verbose=False, cholesky=False)
        # FIXME: spams and lasso
        # FIXED: Don't remember why there's a fixme.
        ss = spams.lasso(np.asfortranarray(target, dtype=np.float64),
                        D=np.asfortranarray(dicttransformed, dtype=np.float64),
                        Q=None, q=None, return_reg_path=False, L=-1, lambda1=self.mu_one / 2.0, lambda2=0.,
                        verbose=False, mode=2)
        # FIXME: task.alpha_matrix and task.policy, which is the one to generate next path?
        # FIXED: They should be the same. Alpha won't be output. So task.policy.
        # FIXME: L*s should be given to both: task.alpha_matrix and task.policy (which I didn't do in matlab code)
        # FIXED: L@s should only be given to initial policies in testing phase.
        s = csc_matrix(ss).toarray()
        task.s = s

        # print("the final s:")
        # print(s)

        # Update A, b and L
        self.A = self.A + np.kron(s @ s.T, hessian)  # (k*d, k*d)
        self.b = self.b + (np.kron(s.T, alpha.T @ hessian)).T  # (k*d, 1)
        ll = (1 / self.T) * np.linalg.inv((1 / self.T) * self.A + self.mu_two * np.eye(self.k * self.d)) @ self.b
        self.L = np.reshape(ll, (self.d, self.k)).T

        # print("the final A:")
        # print(self.A)
        # print("the final b:")
        # print(self.b)
        # print("the final L:")
        # print(self.L)

        return task
