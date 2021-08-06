import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from env import Task
import copy
import random
from PGELLA import Model
from associations import tasks_random
from associations import tasks_lib_normal
from associations import tasks_lib_special
# from associations import task_replace
from associations import pg_rl
from associations import my_plotter
from scipy.io import savemat
import pickle
import time
from TaDeLL import  TaDeLL

def main():

    task = task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
                    cpu_max=random.randint(30, 70), p=0.5, d=2, k=2)

    TaDeLL_Model = TaDeLL(2,3, np.shape(task.feature)[0])

    pg_rl(task,50)
    hessian = task.get_hessian(task.policy['theta'])
    task.hessian_matrix = hessian

    hh = TaDeLL_Model.update(task, task.policy['theta'],task.hessian_matrix)

if __name__ == '__main__':
    main()