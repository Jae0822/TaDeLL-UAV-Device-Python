# This is the file to produce various task, for reshuffling environments purpose


import numpy as np
import copy
import pickle
import random
import math
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


from associations import pg_rl
from env import Task

with open('../mu_sig.pkl', 'rb') as f:
    mu, sig = pickle.load(f)  # Load the mu and sig to extract normalized feature

with open('../TaDeLL_result_k_2.pkl', 'rb') as f:
    means_pg, means_tadell, niter, TaDeLL_Model, tasks0, tasks, testing_tasks, testing_tasks_pg, testing_tasks_TaDeLL = pickle.load(f)

origin = np.array([[0], [0], [0], [0], [0]])


def draw_curves(values_pg, values_TaDeLL):
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(1, len(values_pg) + 1), values_pg, label= 'PG')
    ax.plot(np.arange(1, len(values_TaDeLL) + 1), values_TaDeLL, label= 'TaDeLL')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Ave_Reward')
    ax.set_title("The Ave_Reward for task: " + str(np.linalg.norm(task.feature - origin)))  # Add a title to the axes.
    ax.legend(loc = "best")

# 0. Initialize tasks set
Tasks = []


# 1. Create task
task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
                        cpu_max= random.randint(30, 50), p=0.4 * np.random.random_sample() + 0.3, d=2, k=2)
# task = Task(mean_packet_cycles=random.randint(60, 90), variance_packet_cycles=random.randint(3, 8),
#                         cpu_max= random.randint(100, 150), p=0.4 * np.random.random_sample() + 0.3, d=2, k=2)
task.extract_feature(mu, sig)  # Normalize the plain_feature using mu, sig. self.feature will be updated.


# 2. Evaluate pg_rl and TaDeLL
tsk0 = copy.deepcopy(task)
values_pg = pg_rl(tsk0, niter=50, lr=0.08)

tsk1 = copy.deepcopy(task)
TaDeLL_Model.getDictPolicy_Single(tsk1)  # task.policy['theta'] = np.array(theta_hat)
policy_TaDeLL = tsk1.policy
values_TaDeLL = pg_rl(tsk1, niter=50, lr=0.02)

draw_curves(values_pg, values_TaDeLL)

task.values_pg = values_pg
task.values_TaDeLL = values_TaDeLL
task.policy_TaDeLL = policy_TaDeLL

Tasks.append(task)


# 3. Save task into set and pkl file

with open('Tasks_temp.pkl', 'wb') as f:
    pickle.dump([Tasks], f)


d = 1