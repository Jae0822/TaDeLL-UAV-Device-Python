# To reproduce the curves of TaskList_Set in SourceTask_temp.pkl
# The result is saved in
# https://www.notion.so/4-7_UAV_AC_Positive_Reward-9e98f2705e7f4f0db7b76e279e2c62e0?pvs=4#0c6ef8d446584f3f991c34ebf0402ff8

import numpy as np
import copy
import pickle
import random
import math
import matplotlib.pyplot as plt



with open('../SourceTask_temp.pkl', 'rb') as f:
    TaskList_set, Values_array_set = pickle.load(f)



def draw_curves(values, i):
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(1, len(values) + 1), values, label= 'Task ' + str(i))
    ax.set_xlabel('Steps')
    ax.set_ylabel('Ave_Reward')
    ax.set_title("The Ave_Reward for task" + str(i))  # Add a title to the axes.
    ax.legend(loc = "best")


for i in range(len(TaskList_set)):
    task = TaskList_set[i]
    values = Values_array_set[i]
    draw_curves(values, i)


d = 1