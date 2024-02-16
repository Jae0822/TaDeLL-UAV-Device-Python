import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('TkAgg')
import numpy as np
from env import Task
import copy
import random
# from PGELLA import Model
# from associations import tasks_random
# from associations import tasks_lib_normal
# from associations import tasks_lib_special
# # from associations import task_replace
from associations import pg_rl
# from associations import my_plotter
from scipy.io import savemat
import pickle
import time
from TaDeLL import TaDeLL


def test_feature_function():
    """
    This function is used to test features generation and normalization functions
    tasks.extract_feature()
    TaDeLL.comp_mu_sig()
    :return:
    """
    # Prepare and initialization
    TaDeLL_Model = TaDeLL(2, 3, 5)  # initialize the TaDeLL Model (d, k, m)

    # Step1: Generate a set of tasks randomly
    # with open('tasks_random.pkl', 'rb') as f:
    #     tasks0 = pickle.load(f)

    tasks0 = []
    feature = []
    for i in range(6):
        task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
                    cpu_max=random.randint(30, 70), p=0.4 * np.random.random_sample() + 0.3, d=2, k=2)
        tasks0.append(task)

    # Extract Features for all these tasks
    [mu, sig] = TaDeLL_Model.comp_mu_sig(tasks0)  # Only needs to execute once

    for i in range(0, len(tasks0)):
        task = tasks0[i]
        feature = task.extract_feature(mu, sig)

    return feature


def main():
    """
    This is the function to execute the whole TaDeLL algorithm.
    Step1: Generate 20 training and 10 testing tasks
    Step2: Compute theta for all training tasks
    Step3: Train with TaDeLL.train() and TaDeMTL.train()
    Step4: Testing with testing tasks and trained model
    Step5: Plotting the results
    :return:
    """
    # Prepare and initialization
    niter = 50
    TaDeLL_Model = TaDeLL(2, 2, 5)  # initialize the TaDeLL Model (d, k, m)

    # Step1: Generate 20 training and 10 testing tasks
    with open('Tasks.pkl', 'rb') as f:
        tasks00 = pickle.load(f)  # The task.policy is already the optimal policy
    # for task in tasks0[0]:
    #     task.policy = copy.deepcopy(task.init_policy)

    # Extract and normalize features for all these tasks
    # FIXME: mu and sig obtained through training tasks or the whole set of tasks
    # FIXED: use the whole set of tasks or training tasks doesn't affect the mu, sig a lot
    # [mu, sig] = TaDeLL_Model.comp_mu_sig(tasks0)  # Only needs to execute once
    # for i in range(0, len(tasks0)):
    #     task = tasks0[i]
    #     task.extract_feature(mu, sig)  # Normalize the plain_feature using mu, sig. self.feature will be updated.

    tasks0 = tasks00[0]   # pick the easy ones
    tasks = copy.deepcopy(tasks0[0:20])  # Tasks used for training
    testing_tasks = copy.deepcopy(tasks0[20:40])   # Tasks used for warm start testing

    # Step2: Compute \theta for all training tasks
    for i in range(0, len(tasks)):
        print("compute theta @", i)
        # pg_rl(tasks[i], niter)  # Compute optimal policies for training tasks
        tasks[i].policy['theta'] = tasks[i].policy_pg[-1]
        hessian = tasks[i].get_hessian(tasks[i].policy['theta'])
        tasks[i].hessian_matrix = hessian

    # Step3: Train with TaDeLL.train() and TaDeMTL.train()
    TaDeLL_Model.train(tasks)

    # Step4: Testing with testing tasks and trained model
    testing_tasks_pg = copy.deepcopy(testing_tasks)
    testing_tasks_TaDeLL = copy.deepcopy(testing_tasks)
    rewards_pg = []
    rewards_TaDeLL = []
    for i in range(0, len(testing_tasks)):
        rewards_pg.append([])
        rewards_TaDeLL.append([])

    # Step4.1: PG
    print("start PG test learning")
    for i in range(0, len(testing_tasks_pg)):
        print("testing for PG @", i)
        rewards_pg[i].append(pg_rl(testing_tasks_pg[i], niter))

    # Step4.2: TaDeLL Warm Start
    print("start TaDeLL test learning")
    TaDeLL_Model.getDictPolicy(testing_tasks_TaDeLL)  # get warm start policy for testing tasks
    for i in range(0, len(testing_tasks_TaDeLL)):
        print("testing for TaDeLL @", i)
        rewards_TaDeLL[i].append(pg_rl(testing_tasks_TaDeLL[i], niter))

    means_pg = np.mean(np.concatenate(rewards_pg), 0)
    means_tadell = np.mean(np.concatenate(rewards_TaDeLL), 0)

    # with open('TaDeLL_result_k_3.pkl', 'wb') as f:
    #     pickle.dump([means_pg, means_tadell, niter, TaDeLL_Model, tasks0, tasks, testing_tasks, testing_tasks_pg, testing_tasks_TaDeLL], f)

    with open('TaDeLL_model_k_2_easy_temp.pkl', 'wb') as f:
        pickle.dump([means_pg, means_tadell, niter, TaDeLL_Model, tasks0, tasks, testing_tasks, testing_tasks_pg, testing_tasks_TaDeLL], f)


    # Plotting procedure
    # plt.ion()
    fig, ax = plt.subplots()
    ax.plot(np.arange(niter), means_pg, label='PG')
    ax.plot(np.arange(niter), means_tadell, label='TaDeLL')
    ax.legend()  # Add a legend.
    ax.set_xlabel('Iteration')  # Add an x-label to the axes.
    ax.set_ylabel('Averaged Reward')  # Add a y-label to the axes.
    ax.set_title("Comparison between PG and TaDeLL")  # Add a title to the axes.
    # fig.show()
    # plt.ioff()
    print("Hello Baby")



def pg_task_generation(nTasks):
    """
    This function is used to generate the tasks library
    :return:
    """
    niter = 30
    tasks0 = []
    rewards_pg = []
    values = []
    for i in range(nTasks):
        print("Generate and train task @", i)
        X = False
        while not X:
            task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
                    cpu_max=random.randint(30, 70), p=0.4 * np.random.random_sample() + 0.3, d=2, k=2)
            values = pg_rl(task, niter)

            # plt.ion()
            # fig, ax = plt.subplots()
            # # ax.plot(np.arange(niter), rewards_pg[i][0], label='PG')
            # ax.plot(np.arange(niter), values, label='PG')
            # ax.legend()  # Add a legend.
            # ax.set_xlabel('Iteration')  # Add an x-label to the axes.
            # ax.set_ylabel('Averaged Reward')  # Add a y-label to the axes.
            # ax.set_title("Policy Gradient Learning Curve")  # Add a title to the axes.
            # fig.show()
            # plt.ioff()

            gap = values[-1] - values[0]
            if gap >= 0.8:
                X = True

        task.policy = copy.deepcopy(task.init_policy)

        # Re-evaluate the task
        Y = False
        while not Y:
            values = pg_rl(task, niter)

            # Plotting
            # plt.ion()
            # fig, ax = plt.subplots()
            # # ax.plot(np.arange(niter), rewards_pg[i][0], label='PG')
            # ax.plot(np.arange(niter), values, label='PG')
            # ax.legend()  # Add a legend.
            # ax.set_xlabel('Iteration')  # Add an x-label to the axes.
            # ax.set_ylabel('Averaged Reward')  # Add a y-label to the axes.
            # ax.set_title("Policy Gradient Learning Curve")  # Add a title to the axes.
            # fig.show()
            # plt.ioff()

            gap = values[-1] - values[0]
            if gap >= 0.8:
                Y = True
            else:
                task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
                            cpu_max=random.randint(30, 70), p=0.4 * np.random.random_sample() + 0.3, d=2, k=2)

        rewards_pg.append([])
        rewards_pg[i].append(values)
        tasks0.append(task)

    # for task in tasks0:
    #     task.policy = copy.deepcopy(task.init_policy)

    with open('TaDeLL_Tasks_lib_k_3_temp.pkl', 'wb') as f:
        pickle.dump(tasks0, f)

    print("Tasks generatioin is done")




if __name__ == '__main__':
    main()
    # test_feature_function()
    # pg_task_generation(30)
    d = 1