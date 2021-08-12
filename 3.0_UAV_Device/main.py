'''
This main script includes many functions for different purposes:
#1. Task Generation:
tasks_gen() [simplest way to generate tasks]
main() [test random task generation function: tasks_random(ntasks)]
mainx() [test special tasks]
mainy() [To test and generate qualified tasks]

#2: Debug Basic Functions
main0() [function djd_nac and get_hessian is proved to be correct]
main1() [Test the change of L and A and b when model.update_model is called by the same task repeatedly]
main2() [Test hessian matrix]

#3: Different Training modes (All failed)
mainA1() [Prepare the pg_rl of all tasks for mainA2]
mainA2() [Training Mode A: 50 * 30 + 30 * n]
mainB() [Training Mode B: 50 * 30 * n]
mainC1() [Training Mode C: 50 * 30 * 1 or 50 * 30 * n]
mainC2() [This function is used to compute regular policy gradient and obtain values for tasks used to train in mainC1()]
mainC3() [This function is used to plot and compare results of pg-Ella and regular pg_rl.]
testing() [This function is used to test the performance of PG-ELLA.]

#4: Final Success!
pg_ella() [This function is used to simulate PG-ELLA algorithm as in Matlab: https://github.com/haithamb/pginterella]
'''



import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
from env import Task
import copy
import random
from env import Device
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

# 碎碎念
"""
哎呀，不想工作，好烦好烦，真的好烦。不想工作。明天就是假期，可是我要做PPT好可怜。三天时间够吗？我真的不想工作。
还要写论文。还要写论文。这回写论文没上回那么简单了。真的不想工作。真的好烦。
爸妈回家了，他们会一直待到清明结束，姐也会回家，清明结束了，我的时间又少了一些。
我要做好多事，好多事，主要是这个真的太难了。你知道这个要多久才可以解决吗？你不知道。真的好烦。真的好烦。
唉，真的不想开始工作。今天要做的事情是一个，做完之后仿真又是一个。这个要花多少时间，关键是不一定管用。妈的。
下一个问题几乎是不可解的。妈的。
最大的担心就是不可行可咋办。
刚才上来的时候，外面开始下起了小雨，一点点的，离石今天也在下雨，家里会不会有点冷。
哎呀，在做仿真之前先得把30个任务准备好，准备相似了不行，准备差异大了，容易不收敛。
或许应该随机生成一部分，special cases也准备一部分。
这样一来，LEARNING RATE就需要分别指定，需要建立一个list。
随机生成的部分，要进行一波主动筛选。special cases需要手动指定。
所以需要定义两个函数，一个是自动生成并筛选的函数，一个是手动生成special cases的函数。
自动筛选的标准是什么呢？是增幅？增幅是绝对值还是相对值？
也可以是初始值的绝对大小。
比如只有小于-600或者增幅大于60的。
重新生成新的task，还是只生成新的初始策略。
只要条件不满足，就一直生成，直到满足为止，将新的task append到列表里。
手动生成的special cases，需要指定特别的learning rate。
但同样需要满足50～70的学习步长。
两个函数的结果混在一起，使得lifelong learning可以学习更具有变化性的任务。
"""
"""
还是好烦。上午做了个锤子。进展好慢。
要多少个才算可以呢？五个？十个？先五个吧。
做完之后，还需要把不同函数、不同分类的tasks组合起来，放在一个list里面。
现在只能一个个试。一个个。一个个。一个个。
唉，真的好烦。
外面好冷，实验室还好，宿舍好冷。到处都好冷。我都穿上羽绒服了。

哎呀，太不难过了。为什么有个case怎么也学不到。1000步都会下降。学习速度的大小对于它只是下降快与慢的区别。

失败的尝试。special cases就是不能正常学习。太难过了。是什么的问题呢？
难道是模型本身的参数之间的数值关系吗？
比如CPU不能在某些范围内，否则，
"""
"""
我完蛋了
I'm fucked up.
结果完全不管用。
甚至比一般的还差一点。或者说完全就是跟原来的一摸一样的结果。怎么能说是比一般的好呢？
这个结果好失望。为什么。为什么会这样。为什么。我该怎么办。我要死翘翘。
我望望天空，为什么会发生这种事情。
就是为了让我学一课，所以故意的吗？
When it doesn't work, get away as soon as possible.
Is this the lesson I need to learn?
If it is, I think I already learn hurtfully and tearfully.
"""
'''
2021.08.03
之前担心的问题已经得到解决。解决的过程不用我说有多么惊心动魄了，我也没有完蛋。
只是是在MATLAB里面解决的。
所以现在要在PYTHON里面实现。
幸好我之前已经把PG-ELLA的基础的东西转移到PYTHON里面了，所以现在的工作量没有那么巨大。幸好幸好。
再次谢谢老天爷，帮我渡过人生中目前为止，最为惊险的一次。
谢谢老天爷。
'''


# 碎碎念

def tasks_gen():
    start_time = time.time()
    a = tasks_random(30)
    print("tasks are generated successfully")
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))


def main():
    """
    This function is used to test random task generation function: tasks_random(ntasks)
    """

    niter = 50
    lr = 0.08

    # ntasks = 30
    # tasks = tasks_random(ntasks)

    with open('tasks_random.pkl', 'rb') as f:
        tasks = pickle.load(f)

    x = 13

    task = tasks[x]
    # task.policy["sigma"] = 0.9719

    print("Initial Policy:", task.init_policy["theta"])
    print("task.mean_packet_cycles:", task.mean_packet_cycles)
    print("task.variance_packet_cycles:", task.variance_packet_cycles)
    print("task.cpu_max:", task.cpu_max)

    values_array = pg_rl(task, niter, lr)

    print("rewards_array:", values_array)

    print("Learned Policy:", task.policy["theta"])

    # Plotting procedure
    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(np.arange(niter) + 1, values_array)
    fig.show()
    plt.ioff()
    print("Hello")


def mainx():
    """
    To test special tasks that fits lower lr
    """
    niter = 50
    lr = 0.08

    tasks, learning_rates, niters = tasks_lib_special()

    x = 0

    task = tasks[x]

    lr = learning_rates[x]
    niter = niters[x]

    print("Initial Policy:", task.policy["theta"])
    print("task.mean_packet_cycles:", task.mean_packet_cycles)
    print("task.variance_packet_cycles:", task.variance_packet_cycles)
    print("task.cpu_max:", task.cpu_max)

    print("learning rate:", lr)
    print("niter:", niter)

    values_array = pg_rl(task, niter, lr)

    print("rewards_array:", values_array)

    print("Learned Policy:", task.policy["theta"])

    # Plotting procedure
    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(np.arange(niter) + 1, values_array)
    fig.show()
    plt.ioff()
    print("Hello")


def mainy():
    """
    To test and generate qualified tasks
    """
    niter = 50
    lr = 0.08

    task = Task(mean_packet_cycles=24, variance_packet_cycles=8, cpu_max=44, p=0.5, d=2, k=2)
    # task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8), cpu_max=random.randint(30, 70), p=0.5, d=2, k=2)

    task.init_policy["theta"] = np.array([[0.09146959], [0.03107515]])
    task.policy = copy.deepcopy(task.init_policy)

    print("Initial Policy:", task.policy)
    print("task.mean_packet_cycles:", task.mean_packet_cycles)
    print("task.variance_packet_cycles:", task.variance_packet_cycles)
    print("task.cpu_max:", task.cpu_max)

    values_array = pg_rl(task, niter, lr)

    print("rewards_array:", values_array)

    print("Learned Policy:", task.policy)

    # Plotting procedure
    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(np.arange(niter) + 1, values_array)
    fig.show()
    plt.ioff()
    print("Hello")


def main0():
    """
    function djd_nac and get_hessian is proved to be correct
    Compare the results of these two functions with corresponding matlab functions (using the same data/path as input)
    """
    # task = Task(mean_packet_cycles=10, variance_packet_cycles=3, cpu_max=50, p=0.5, d=2, k=2)
    # path = task.collect_path()
    # djd_theta = task.djd_nac(path)
    # task.get_hessian(path)
    # print("djd_theta:", djd_theta)
    # print("policy:", task.policy['theta'])
    # print("hessian:", task.hessian_matrix)
    # file_name = 'data_py.mat'
    # savemat(file_name, {'path_py':path, 'hessian_py': task.hessian_matrix})

    # task = Task(mean_packet_cycles=10, variance_packet_cycles=3, cpu_max=50, p=0.5, d=2, k=2)
    # task.policy["theta"] = task.policy["theta"] + djd_theta
    # print("policy:", task.policy["theta"] )
    # values = []
    # values.append(task.get_value(path))
    # print("rewards:", values)
    # values_array = np.array(values)
    # print("rewards_array:", values_array)
    # fig, ax = plt.subplots()

    """
    PG-ELLA algorithm training process
    """
    niter = 50

    tasks0 = []
    tasks0 = tasks_lib_normal(tasks0)
    tasks = [tasks0[0], tasks0[4], tasks0[8]]  # Picking the task I want

    model = Model(d=2, k=2)

    values = []
    my_list = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    for i in my_list:
        task = tasks[i]

        print("task: ", i)

        values.append(pg_rl(task, niter))
        print("task.policy:", task.policy["theta"])
        print("values of pg_rl:", values[-1][-1])

        task.get_hessian()
        task = model.update_model(task)

        task.policy["theta"] = model.L @ task.s
        print("model.policy:", model.L @ task.s)
        print("values of PG-ELLA:", task.get_value())

        print("hello")

    # Generate the devices in a (1km X 1km) square field.
    # num_sys = 5  # Number of devices in the system
    # devices = []
    # for i in range(num_sys):
    #     devices = devices.append(Device())
    """
    PG-ELLA testing process
    """


def main1():
    """
    Test the change of L and A and b when model.update_model is called by the same task repeatedly
    Three stages will be experienced in the model.update_model function:
    original (the one from last update: s, A, b L),
    revised (A, b, L, A and b are subtracted, L is reinitialized),
    final (A and b are added again, L is obtained from A and b).
    (1) The results show the revised A and b can be all zeros matrix again. (If the same alpha and hessian are given)
    However, because the L is changed, so the s obtained from Lasso is changed too.
    In accordance, the new A, b and L can be different in each iteration.
    (2) When the same task is called by model.update_model function repeatedly,
    the L and A and b are as small as 10e-07.
    There's always a column of L becomes all zeros.
    (3) For more tasks, the L will increase gradually.
    """
    niter = 50

    tasks0 = []
    tasks0 = tasks_lib_normal(tasks0)
    tasks = [tasks0[0], tasks0[4], tasks0[8]]  # Picking the task I want

    model = Model(d=2, k=2)

    values = []
    # my_list = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    my_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]

    print("task: ", 0)

    values.append(pg_rl(tasks[0], niter))
    print("task.policy:", tasks[0].policy["theta"])
    print("values of pg_rl:", values[-1][-1])

    alpha = tasks[0].policy["theta"]
    hessian = tasks[0].get_hessian(alpha)
    tasks[0].hessian_matrix = hessian

    j = 0
    for i in my_list:
        print("********************************")
        print("Iteration:", j)
        j = j + 1

        task = tasks[i]

        task = model.update_model(task, alpha, hessian)

        # task.policy["theta"] = model.L @ task.s
        print("model.policy:", model.L @ task.s)
        print("values of PG-ELLA:", task.get_value(model.L @ task.s))

        print("hello")


def main2():
    """
    Test hessian matrix:
    The result shows for the same alpha policy, the result of function get_hessian(alpha) can be slightly different.
    :return:
    """
    niter = 50

    tasks0 = []
    tasks0 = tasks_lib_normal(tasks0)
    tasks = [tasks0[0], tasks0[4], tasks0[8]]  # Picking the task I want

    values = []
    # my_list = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    my_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]

    print("task: ", 0)

    values.append(pg_rl(tasks[0], niter))
    print("task.policy:", tasks[0].policy["theta"])
    print("values of pg_rl:", values[-1][-1])

    alpha = tasks[0].policy["theta"]
    hessian = tasks[0].get_hessian(alpha)  # debug and run this line in debugger console
    tasks[0].hessian_matrix = hessian


def mainA1():
    """
    Prepare the pg_rl of all tasks for mainA2.
    :return:
    """
    start_time = time.time()

    niter = 50
    lr = 0.08

    # ntasks = 30
    # tasks = tasks_random(ntasks)

    # TODO: Use tasks randomly generated. Need to decide if the learning process needs to repeat
    with open('tasks_random.pkl', 'rb') as f:
        tasks = pickle.load(f)

    values = []
    alphas = []
    hessians = []
    for i in range(0, np.shape(tasks)[0]):
        task = tasks[i]
        values.append(pg_rl(task, niter, lr))

        alpha = task.policy["theta"]
        hessian = task.get_hessian(alpha)
        task.hessian_matrix = hessian

        alphas.append(alpha)
        hessians.append(hessian)

        print("task:", i)
        print("task.policy:")
        print(task.policy["theta"])
        print("values of pg_rl:")
        print(values[-1][-1])

    with open('PlanA_values_alphas_hessians.pkl', 'wb') as f:
        pickle.dump([values, alphas, hessians], f)

    print("tasks are generated successfully")
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))


def mainA2():
    """
    Training Mode A: 50 * 30 + 30 * n
    (1) Full training for one time.
    (2) Half training for n times. Each half training contains a pg_ella update. (Using the same result from step (1))
    """

    # ntasks = 30
    # tasks = tasks_random(ntasks)

    with open('tasks_random.pkl', 'rb') as f:
        tasks = pickle.load(f)

    with open('PlanA_values_alphas_hessians.pkl', 'rb') as f:
        values, alphas, hessians = pickle.load(f)

    model = Model(d=2, k=2)

    list_single = list(range(0, np.shape(tasks)[0]))

    learned_values = []
    for x in range(0, np.shape(tasks)[0]):
        learned_values.append([])

    for j in range(0, 10):
        print("Iteration:", j)

        for i in list_single:
            task = tasks[i]

            print("task: ", i)

            task = model.update_model(task, alphas[i], hessians[i])

            learned_values[i].append(task.get_value(model.L @ task.s))

            print("model.policy:")
            print(model.L @ task.s)
            print("values of PG-ELLA:")
            print(task.get_value(model.L @ task.s))

    with open('PlanA_learned_values_model.pkl', 'wb') as f:
        pickle.dump([learned_values, model], f)

    print("hello")


def mainB():
    """
    Training Mode B: 50 * 30 * n
    full training for n times. Each full training contains a full pg_rl
    """
    niter = 50

    tasks0 = []
    tasks0 = tasks_lib_normal(tasks0)
    tasks = [tasks0[0], tasks0[4], tasks0[8]]  # Picking the task I want

    values = []
    # for i in range(0, np.shape(tasks)[0]):
    #
    #     task = tasks[i]
    #     values.append(pg_rl(task, niter))
    #
    #     print("task:", i)
    #     print("task.policy:")
    #     print(task.policy["theta"])
    #     print("values of pg_rl:")
    #     print(values[-1][-1])

    model = Model(d=2, k=2)

    list_single = list(range(0, np.shape(tasks)[0]))
    my_list = list_single + list_single + list_single

    for i in my_list:
        task = tasks[i]

        print("task: ", i)

        values.append(pg_rl(task, niter))
        print("task.policy:")
        print(task.policy["theta"])
        print("values of pg_rl:")
        print(values[-1][-1])

        hessian = task.get_hessian(task.policy["theta"])
        task.hessian_matrix = hessian

        task = model.update_model(task, task.policy["theta"], task.hessian_matrix)

        task.policy["theta"] = model.L @ task.s
        print("model.policy:")
        print(model.L @ task.s)
        print("values of PG-ELLA:")
        print(task.get_value(task.policy["theta"]))

        print("hello")


def mainC1():
    """
    Training Mode C: 50 * 30 * 1 or 50 * 30 * n
    No full pg_rl.
    Similar to mode B, except each full training contains one step of pg_rl.
    """
    start_time = time.time()

    niter = 1

    with open('tasks_random.pkl', 'rb') as f:
        tasks0 = pickle.load(f)

    # tasks = tasks0[0: 3]
    tasks = []
    tasks.append(tasks0[2])
    tasks.append(tasks0[12])
    tasks.append(tasks0[18])
    tasks.append(tasks0[19])
    tasks.append(tasks0[20])
    tasks.append(tasks0[22])

    model = Model(d=2, k=2)

    values_pg_rl = []
    values_pg_ella = []
    for x in range(0, np.shape(tasks)[0]):
        values_pg_rl.append([])
        values_pg_ella.append([])

    list_single = list(range(0, np.shape(tasks)[0]))

    for i in range(0, 50):

        for j in list_single:
            task = tasks[j]

            print("task: ", j)

            array_value = pg_rl(task, niter)
            values_pg_rl[j].append(array_value[0])
            print("task.policy:")
            print(task.policy["theta"])
            print("values of pg_rl:")
            print(values_pg_rl[j][-1])

            hessian = task.get_hessian(task.policy["theta"])
            task.hessian_matrix = hessian

            task = model.update_model(task, task.policy["theta"], task.hessian_matrix)

            task.policy["theta"] = model.L @ task.s
            values_pg_ella[j].append(task.get_value(task.policy["theta"]))
            print("model.policy:")
            print(model.L @ task.s)
            print("values of PG-ELLA:")
            print(values_pg_ella[j][-1])

    print("hello")

    values_pg_rl = np.array(values_pg_rl)
    values_pg_ella = np.array(values_pg_ella)

    with open('PlanC1_training_values_sample.pkl', 'wb') as f:
        pickle.dump([values_pg_rl, values_pg_ella, model], f)

    return start_time


def mainC2():
    """
    This function is used to compute regular policy gradient and obtain values for tasks used to train in mainC1()
    :return:
    """
    niter = 50

    with open('tasks_random.pkl', 'rb') as f:
        tasks0 = pickle.load(f)

    # tasks = tasks0[0 : 3]

    tasks = []
    tasks.append(tasks0[2])
    tasks.append(tasks0[12])
    tasks.append(tasks0[18])
    tasks.append(tasks0[19])
    tasks.append(tasks0[20])
    tasks.append(tasks0[22])

    regular_values = []
    for x in range(0, np.shape(tasks)[0]):
        regular_values.append([])

    for i in range(0, len(tasks)):
        task = tasks[i]

        regular_values[i].append(pg_rl(task, niter))

    regular_values = np.array(regular_values)

    # plt.ion()
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # my_plotter(ax1, np.arange(niter) + 1, regular_values[0][0], {'marker': 'o'})
    # my_plotter(ax2, np.arange(niter) + 1, regular_values[1][0], {'marker': 'o'})
    # my_plotter(ax3, np.arange(niter) + 1, regular_values[2][0], {'marker': 'o'})
    # fig.show()
    # plt.ioff()
    # print("Hello")

    with open('PlanC2_regular_values_sample.pkl', 'wb') as f:
        pickle.dump(regular_values, f)


def mainC3(start_time):
    """
    This function is used to plot and compare results of pg-Ella and regular pg_rl.
    :return:
    """
    with open('PlanC1_training_values_sample.pkl', 'rb') as f:
        values_pg_rl, values_pg_ella, model = pickle.load(f)

    with open('PlanC2_regular_values_sample.pkl', 'rb') as f:
        regular_values = pickle.load(f)

    niter = 50

    for i in range(len(regular_values)):
        plt.ion()
        fig, ax = plt.subplots()
        ax.plot(np.arange(niter) + 1, regular_values[i][0], label='regular PG')
        ax.plot(np.arange(niter) + 1, values_pg_rl[i], label='values_pg_rl')
        ax.plot(np.arange(niter) + 1, values_pg_ella[i], label='values_pg_ella')
        plt.legend(loc='best')
        fig.show()
        plt.ioff()
        print("Hello")

    print("--- %s minutes ---" % ((time.time() - start_time) / 60))


def testing():
    """
    This function is used to test the performance of PG-ELLA.
    :return:
    """
    niter = 1
    lr = 0.08

    navg = 50

    task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
                cpu_max=random.randint(30, 70), p=0.5, d=2, k=2)

    values_pg_rl = []
    alphas = []
    hessians = []
    for i in range(0, navg):
        values_pg_rl.append(pg_rl(task, niter, lr))
        alpha = task.policy["theta"]
        alphas.append(alpha)
        hessians.append(task.get_hessian(alpha))
        task.policy = copy.deepcopy(task.init_policy)

    value_pg_rl = np.mean(values_pg_rl)
    print("pg_rl:", value_pg_rl)

    alpha_00 = np.mean([alphas[x][0][0] for x in range(len(alphas))])
    alpha_10 = np.mean([alphas[x][1][0] for x in range(len(alphas))])
    alpha_avg = np.array([[alpha_00], [alpha_10]])

    hessian_00 = np.mean([hessians[x][0][0] for x in range(len(hessians))])
    hessian_01 = np.mean([hessians[x][0][1] for x in range(len(hessians))])
    hessian_10 = np.mean([hessians[x][1][0] for x in range(len(hessians))])
    hessian_11 = np.mean([hessians[x][1][1] for x in range(len(hessians))])
    hessian_avg = np.array([[hessian_00, hessian_01], [hessian_10, hessian_11]])

    with open('PlanC1_training_values.pkl', 'rb') as f:
        values_pg_rl, values_pg_ella, model = pickle.load(f)

    values_pg_ella = []
    for i in range(0, navg):
        task = model.update_model(task, task.init_policy["theta"], task.get_hessian(task.init_policy["theta"]))
        # task = model.update_model(task, alpha_avg, hessian_avg)
        values_pg_ella.append(task.get_value(model.L @ task.s))
    value_pg_ella = np.mean(values_pg_ella)
    print("pg_ella:", value_pg_ella)


def pg_ella():
    '''
    2021-08-03
    This function is used to simulate PG-ELLA algorithm as in Matlab
    Step 1: Generate 3 or 20 tasks
    Step 2: Learn with PG and obtain \alpha
    Step 3: Learn with PG-ELLA
    Step 4: Compare both results
    :return:
    '''

    # Step 1: Generate 3 to 20 tasks
    niter = 50
    with open('tasks_random.pkl', 'rb') as f:
        tasks0 = pickle.load(f)
    # tasks = tasks0[0 : 3]
    tasks = []
    tasks.append(tasks0[2])
    tasks.append(tasks0[12])
    tasks.append(tasks0[18])
    tasks.append(tasks0[19])
    tasks.append(tasks0[20])
    tasks.append(tasks0[22])

    values_pg_rl = []
    values_pg_ella = []
    values_pg_train = []
    for x in range(0, np.shape(tasks)[0]):
        values_pg_rl.append([])
        values_pg_ella.append([])
        values_pg_train.append([])

    # Step 2: Learn with PG and obtain \alpha
    for i in range(0, len(tasks)):
        # task = tasks[i]
        values_pg_rl[i].append(pg_rl(tasks[i], niter))

    regular_values = np.array(values_pg_rl)
    means_regular = np.mean(np.concatenate(regular_values), 0)
    print("pg_rl:", means_regular)

    # Step 3: Learn with PG-ELLA
    tasks_pre = copy.deepcopy(tasks)  # Prepare tasks for PG - ELLA
    observe_flag = 0
    observed_tasks = np.zeros(np.shape(tasks_pre)[0])
    counter = 1
    LimOne = 0
    LimTwo = np.shape(tasks_pre)[0]
    model = Model(d=2, k=2)

    while not observe_flag:
        # choose a task
        if np.all(observed_tasks):
            observe_flag = 1
            print('All tasks have been observed')

        taskID = np.random.randint(LimOne, LimTwo)
        task = tasks_pre[taskID]

        if observed_tasks[taskID] == 0:
            observed_tasks[taskID] = 1
            model.T = model.T + 1

        values_pg_train[taskID].append(pg_rl(task, 1)[0]) # 执行一步policy gradient更新

        # prepare the task hessian and alpha
        hessian = task.get_hessian(task.policy["theta"])
        task.hessian_matrix = hessian

        # update model
        task = model.update_model(task, task.policy["theta"], task.hessian_matrix)

        counter = counter + 1

    print('training process finishes')
    
    #Testing Phase
    tasks_PG_Test = copy.deepcopy(tasks)
    tasks_PGELLA_Test = copy.deepcopy(tasks_pre)
    values_pg = []
    values_pg_ella = []
    for i in range(0, len(tasks_PG_Test)):
        tasks_PG_Test[i].policy = {'theta': np.random.rand(2, 1), 'sigma': 5}  # 重新生成新的随机初始向量
        tasks_PGELLA_Test[i].policy['theta'] = model.L @ tasks_PGELLA_Test[i].s  # 初始策略来自于PG-ELLA算法
        values_pg.append([])
        values_pg_ella.append([])

    for i in range(0, len(tasks_PG_Test)):
        values_pg[i].append(pg_rl(tasks_PG_Test[i], niter))
        values_pg_ella[i].append(pg_rl(tasks_PGELLA_Test[i], niter))
    print('Testing process finishes')

    means_pg = np.mean(np.concatenate(values_pg), 0)
    means_pgella = np.mean(np.concatenate(values_pg_ella), 0)

    # Plotting procedure
    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(np.arange(niter), means_pg, label='PG')
    ax.plot(np.arange(niter), means_pgella, label='PG-ELLA')
    ax.legend()  # Add a legend.
    ax.set_xlabel('Iteration')  # Add an x-label to the axes.
    ax.set_ylabel('Averaged Reward')  # Add a y-label to the axes.
    ax.set_title("Comparison between PG and PG-ELLA")  # Add a title to the axes.
    fig.show()
    plt.ioff()
    print("Hello Baby")

    with open('Final_Result.pkl', 'wb') as f:
        pickle.dump([tasks, tasks_pre, means_pg, means_pgella, niter], f)

    return tasks, tasks_pre, means_pg, means_pgella


def plotting():

    with open('Final_Result.pkl', 'rb') as f:
        tasks, tasks_pre, means_pg, means_pgella, niter = pickle.load(f)

    # Plotting procedure
    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(np.arange(niter), means_pg, label='PG')
    ax.plot(np.arange(niter), means_pgella, label='PG-ELLA')
    ax.legend()  # Add a legend.
    ax.set_xlabel('Iteration')  # Add an x-label to the axes.
    ax.set_ylabel('Averaged Reward')  # Add a y-label to the axes.
    ax.set_title("Comparison between PG and PG-ELLA")  # Add a title to the axes.
    fig.show()
    plt.ioff()
    print("Hello Baby")


if __name__ == '__main__':
    # start_time = mainC1()
    # mainC2()
    # mainC3(start_time)
    pg_ella()
    # plotting()

