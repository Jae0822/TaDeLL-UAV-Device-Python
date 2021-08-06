import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pickle
import numpy as np
from env import Task
import copy
import random
from env import Device
from PGELLA import Model



def pg_rl(task, niter=50, lr=0.08):
    """
    Basic policy gradient learning can work fine for env.Task
    """
    # niter = 50
    values = []
    for i in range(niter):  # Iterate policy gradient process
        path = task.collect_path(task.policy["theta"])

        djd_theta = task.djd_nac(path)
        task.policy["theta"] = task.policy["theta"] + lr * djd_theta

        values.append(task.get_value(task.policy["theta"]))

        # print(i)
        # print("djd_theta", djd_theta, "policy:", task.policy["theta"], "rewards:", values[-1])
    values_array = np.array(values)

    # print("rewards_array:", values_array)
    # # Plotting procedure
    # plt.ion()
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(niter) + 1, values_array)
    # fig.show()
    # plt.ioff()
    # print("Hello")

    return values_array


def tasks_lib_normal():
    """
    The mean_packet_cycles and variance_packet_cycles decides a unique learning pattern.
    The initial policy decides the initial value.
    The CPU_max decides the upper limit can achieve. (task4 and task5)
    """
    tasks = []

    # 1:
    tasks.append(Task(mean_packet_cycles=20, variance_packet_cycles=4, cpu_max=50, p=0.5, d=2, k=2))
    tasks[-1].init_policy["theta"] = np.array([[0.68961307], [0.80623155]])  # -384.94939997, -312.631174871 lr = 0.1

    tasks.append(Task(mean_packet_cycles=20, variance_packet_cycles=4, cpu_max=50, p=0.5, d=2, k=2))
    tasks[-1].init_policy["theta"] = np.array([[0.36685647], [0.22691777]])  # -354.86392837, -314.70760995 lr = 0.1

    tasks.append(Task(mean_packet_cycles=20, variance_packet_cycles=4, cpu_max=50, p=0.5, d=2, k=2))
    tasks[-1].init_policy["theta"] = np.array([[0.34329939], [0.02977314]])  # 652.72300486, -322.76500375, lr =.08

    # 2:
    tasks.append(Task(mean_packet_cycles=15, variance_packet_cycles=4, cpu_max=30, p=0.5, d=2, k=2))
    tasks[-1].init_policy["theta"] = np.array([[0.23183801], [0.01636072]])  # -300.2785963, -245.09513246 lr = 0.03 ～ 0.06

    tasks.append(Task(mean_packet_cycles=15, variance_packet_cycles=4, cpu_max=30, p=0.5, d=2, k=2))
    tasks[-1].init_policy["theta"] = np.array([[0.58992776], [0.78118324]])  # -260.514479, -245.20828643 lr = 0.03 ~ 0.08

    tasks.append(Task(mean_packet_cycles=15, variance_packet_cycles=4, cpu_max=30, p=0.5, d=2, k=2))
    tasks[-1].init_policy["theta"] = np.array([[0.01992776], [0.2118324]])  # -343.31292683, -243.13323323 lr = 0.08

    # 3:
    tasks.append(Task(mean_packet_cycles=25, variance_packet_cycles=4, cpu_max=30, p=0.5, d=2, k=2))
    tasks[-1].init_policy["theta"] = np.array([[0.89650231], [0.67431754]])  # -490.45086464, -413.62323275 lr = 0.08 ~ 0.2

    tasks.append(Task(mean_packet_cycles=25, variance_packet_cycles=4, cpu_max=30, p=0.5, d=2, k=2))
    tasks[-1].init_policy["theta"] = np.array([[0.64279542], [0.16876687]])  # -435.87392534, -392.11129844 lr = 0.08 ~ 0.2

    # 4:
    tasks.append(Task(mean_packet_cycles=25, variance_packet_cycles=4, cpu_max=50, p=0.5, d=2, k=2))
    tasks[-1].init_policy["theta"] = np.array([[0.59851199], [0.32482745]])  # -410.3782683, -400.57338816 lr = 0.08

    tasks.append(Task(mean_packet_cycles=25, variance_packet_cycles=4, cpu_max=50, p=0.5, d=2, k=2))
    tasks[-1].init_policy["theta"] = np.array([[0.08422641], [0.91186847]])  # -591.1078398, -400.92615604 lr = 0.08 ~ 0.1

    # 5:
    tasks.append(Task(mean_packet_cycles=25, variance_packet_cycles=4, cpu_max=70, p=0.5, d=2, k=2))
    tasks[-1].init_policy["theta"] = np.array([[0.59851199], [0.32482745]])  # -593.05333301, -402.50255933 lr = 0.08

    # 6:
    tasks.append(Task(mean_packet_cycles=34, variance_packet_cycles=8, cpu_max=70, p=0.5, d=2, k=2))
    tasks[-1].init_policy["theta"] = np.array([[0.75295495], [0.83296109]])  # -1135.46952025, -626.43361927 lr = 0.1

    # [x.policy = copy.deepcopy(x.init_policy) for x in tasks]
    for i in range(len(tasks)):
        tasks[i].policy = copy.deepcopy(tasks[i].init_policy)

    return tasks


def tasks_lib_special():
    """
    This function is used to manually generate special tasks that can't be learned by lr = 0.1.
    :return:
    """

    tasks = []
    learning_rates = []
    niters = []

    # 0:
    tasks.append(Task(mean_packet_cycles=34, variance_packet_cycles=8, cpu_max=70, p=0.5, d=2, k=2))
    tasks[-1].init_policy["theta"] = np.array([[0.09146959], [0.03107515]])  # lr = 0.01
    learning_rates.append(0.01)  # -926.82793319, -843.11681483
    niters.append(200)

    # 1:
    tasks.append(Task(mean_packet_cycles=24, variance_packet_cycles=8, cpu_max=44, p=0.5, d=2, k=2))
    tasks[-1].init_policy["theta"] = np.array([[0.09146959], [0.03107515]])  # lr = 0.005
    learning_rates.append(0.001)  # -837.93793494, -710.33229272  -682.81793177  -633.84973046 -585.79895919
    niters.append(1000)

    for i in range(len(tasks)):
        tasks[i].policy = copy.deepcopy(tasks[i].init_policy)

    return tasks, learning_rates, niters


def tasks_random(ntask):
    tasks = []
    niter = 50
    lr = 0.08
    for i in range(ntask):
        task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
                    cpu_max=random.randint(30, 70), p=0.5, d=2, k=2)
        values_array = pg_rl(task, niter, lr)
        # init_value = values_array[0]
        gap = values_array[-1] - values_array[0]
        while gap < 60:
            task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
                        cpu_max=random.randint(30, 70), p=0.5, d=2, k=2)
            values_array = pg_rl(task, niter, lr)  # The policy is changed during learning.
            # init_value = values_array[0]
            gap = values_array[-1] - values_array[0]
        tasks.append(task)
    for task in tasks:
        task.policy = copy.deepcopy(task.init_policy)


    # Cross validation
    # Once a task index is removed from X, the index x will jump to the one after the next.
    # The remove process goes alternately. Until X becomes empty.
    X = list(range(0, np.shape(tasks)[0]))
    while X:
        for x in X:
            task = tasks[x]
            values_array = pg_rl(task, niter, lr)
            gap = values_array[-1] - values_array[0]
            if gap >= 60:
                X.remove(x)
            else:
                task_replace(tasks, x)

    for task in tasks:
        task.policy = copy.deepcopy(task.init_policy)

    with open('tasks_random.pkl', 'wb') as f:
        pickle.dump(tasks, f)

    return tasks


def task_replace(tasks, x):
    niter = 50
    lr = 0.08

    task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
                cpu_max=random.randint(30, 70), p=0.5, d=2, k=2)
    values_array = pg_rl(task, niter, lr)
    # init_value = values_array[0]
    gap = values_array[-1] - values_array[0]
    while gap < 60:
        task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
                    cpu_max=random.randint(30, 70), p=0.5, d=2, k=2)
        values_array = pg_rl(task, niter, lr)  # The policy is changed during learning.
        # init_value = values_array[0]
        gap = values_array[-1] - values_array[0]
    task.policy = copy.deepcopy(task.init_policy)
    tasks[x] = task

def my_plotter(ax, data1, data2, param_dict):
    """
    A helper function to make a graph

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    data1 : array
       The x data

    data2 : array
       The y data

    param_dict : dict
       Dictionary of kwargs to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
    out = ax.plot(data1, data2, **param_dict)
    return out