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
from TaDeLL import TaDeLL



def main():
    # Step 1: read TaDeLL model
    with open('TaDeLL_result_k_2.pkl', 'rb') as f:
        means_pg, means_tadell, niter, TaDeLL_Model, tasks0, tasks, testing_tasks, testing_tasks_pg, testing_tasks_TaDeLL = pickle.load(
            f)
    # Step 2: read task file
    with open('SourceTask_temp.pkl', 'rb') as f:
        TaskList_set, Values_array_set = pickle.load(f)
    task_pg = TaskList_set[1]
    task_TaDeLL = copy.deepcopy(task_pg)

    # Step 3: get pg and TaDeLL policy separately
    niter = 50
    rewards_pg = []
    rewards_TaDeLL = []
    rewards_pg.append(pg_rl(task_pg, niter))
    TaDeLL_Model.getDictPolicy(task_TaDeLL)
    rewards_TaDeLL.append(pg_rl(task_TaDeLL, niter))


    # Step 4: record values and plotting
    with open('temp.pkl', 'wb') as f:
        pickle.dump([rewards_pg, rewards_TaDeLL, niter, TaDeLL_Model, task_pg, task_TaDeLL], f)

    # Plotting procedure
    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(np.arange(niter), rewards_pg[0], label='PG')
    ax.plot(np.arange(niter), rewards_TaDeLL[0], label='TaDeLL')
    ax.legend()  # Add a legend.
    ax.set_xlabel('Iteration')  # Add an x-label to the axes.
    ax.set_ylabel('Averaged Reward')  # Add a y-label to the axes.
    ax.set_title("Comparison between PG and TaDeLL")  # Add a title to the axes.
    fig.show()
    plt.ioff()
    print("Hello Baby")


if __name__ == '__main__':
    main()
