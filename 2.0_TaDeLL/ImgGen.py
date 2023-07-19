"""
画图：
Fig1:单个MDP的WARM-START与REGULAR POLICY Gradient对比图
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from statistics import mean


# 可以查看pkl文件的内容
# f = open('TaDeLL_result_k_3.pkl','rb')
# data = pickle.load(f)
# print(data)
# len(data)


with open('TaDeLL_result_k_3.pkl', 'rb') as f:
    [means_pg, means_tadell, niter, TaDeLL_Model, tasks0, tasks, testing_tasks, testing_tasks_pg,
     testing_tasks_TaDeLL] = pickle.load(f)

# Plotting procedure
plt.ion()
fig, ax = plt.subplots()
ax.plot(np.arange(niter), means_pg, label='PG')
# ax.plot(np.arange(niter), means_pgella, label='PG-ELLA')
ax.plot(np.arange(niter), means_tadell, label='ZSLL')
ax.legend()  # Add a legend.
ax.set_xlabel('Iteration')  # Add an x-label to the axes.
ax.set_ylabel('Averaged Reward')  # Add a y-label to the axes.
ax.set_title("Comparison between PG and ZSLL")  # Add a title to the axes.

plt.savefig('Figure_temp.eps', format='eps')


d = 1