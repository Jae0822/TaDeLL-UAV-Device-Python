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
from scipy.spatial.distance import pdist

from associations import pg_rl_values_policies_gradients
from env import Task


with open('../mu_sig.pkl', 'rb') as f:
    mu, sig = pickle.load(f)  # Load the mu and sig to extract normalized feature

with open('../TaDeLL_result_k_2.pkl', 'rb') as f:
    means_pg, means_tadell, niter, TaDeLL_Model, tasks0, tasks, testing_tasks, testing_tasks_pg, testing_tasks_TaDeLL = pickle.load(f)

origin = np.array([[0], [0], [0], [0], [0]])
origin_cos = np.array([[1], [1], [1], [1], [1]])

def draw_curves(values_pg, values_TaDeLL):
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(1, len(values_pg) + 1), values_pg, label= 'PG')
    ax.plot(np.arange(1, len(values_TaDeLL) + 1), values_TaDeLL, label= 'TaDeLL')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Ave_Reward')
    ax.set_title("The Ave_Reward for task: " + str(np.linalg.norm(task.feature - origin)))  # Add a title to the axes.
    ax.legend(loc = "best")

# 0. Initialize tasks set
# Tasks = []


# 1. Create task
task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
                        cpu_max= random.randint(30, 50), p=0.4 * np.random.random_sample() + 0.3, d=2, k=2)
# task = Task(mean_packet_cycles=random.randint(60, 90), variance_packet_cycles=random.randint(3, 8),
#                         cpu_max= random.randint(100, 150), p=0.4 * np.random.random_sample() + 0.3, d=2, k=2)
task.extract_feature(mu, sig)  # Normalize the plain_feature using mu, sig. self.feature will be updated.


# 2. Evaluate pg_rl and TaDeLL
tsk0 = copy.deepcopy(task)
values_pg, policies_pg, gradients_pg = pg_rl_values_policies_gradients(tsk0, niter=50, lr=0.08)

tsk1 = copy.deepcopy(task)
TaDeLL_Model.getDictPolicy_Single(tsk1)  # task.policy['theta'] = np.array(theta_hat)
policy_TaDeLL = tsk1.policy
values_TaDeLL, policies_TaDeLL, gradients_TaDeLL = pg_rl_values_policies_gradients(tsk1, niter=50, lr=0.02)

draw_curves(values_pg, values_TaDeLL)

# 重要的原始属性
# task.init_policy
# task.plain_feature
# task.feature
# 记录pg-rl和TaDeLL的values，policies，gradients的记录
task.values_pg = values_pg
task.values_TaDeLL = values_TaDeLL
task.policy_pg =policies_pg
task.policy_TaDeLL = policies_TaDeLL
task.gradients_pg = gradients_pg
task.gradients_TaDeLL = gradients_TaDeLL
# 一些人工特征提取
# task.cost_gap = values_TaDeLL[0] - values_pg[0]
# task.first_gradient =
task.distance = np.linalg.norm(task.feature - origin)
task.cos_distance = 1 - np.dot(np.squeeze(task.feature), np.squeeze(origin_cos)) / (np.linalg.norm(np.squeeze(task.feature)) * np.linalg.norm(np.squeeze(origin_cos)))
# 下面的计算方法可以得到一样的结果
# task.cos_distance = pdist(np.vstack([np.squeeze(task.feature), np.squeeze(origin_cos)]), 'cosine')



# 3. Save task into set and pkl file
# check the task and save it when it's good
task.convergence_steps = 1 # 手动输入

with open('Tasks_temp.pkl', 'rb') as f:
    Tasks = pickle.load(f)

Tasks[0].append(task)
Tasks[1].append(task)


with open('Tasks_temp.pkl', 'wb') as f:
    pickle.dump(Tasks, f)


d = 1