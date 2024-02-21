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



    # Type 1: All easy
    # tasks0 = tasks00[0]   # pick the easy ones
    # Type 2: All difficult
    # tasks0 = tasks00[1]   # pick the diffucult ones
    #  l + g = [0:36]
    # for difficult ones: l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 20, 21, 22, 23, 24, 26, 28, 29, 33, 34, 36]
    # those not good: g = [15, 18, 19, 25, 27, 30, 31, 32, 35]
    # l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 20, 21, 22, 23, 24, 26, 28, 29, 33, 34, 36]
    # tasks0 = [tasks00[1][x] for x in l]
    # tasks = copy.deepcopy(tasks0[0:20])  # Tasks used for training
    # testing_tasks = copy.deepcopy(tasks0[20:-1])   # Tasks used for warm start testing
    # Type 3: Half easy half difficult
    l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 20, 21, 22, 23, 24, 26, 28, 29, 33, 34, 36]
    tasks_easy = tasks00[0]
    tasks_difficult = [tasks00[1][x] for x in l]
    tasks0 = tasks_easy[0:10] + tasks_difficult[0:10] + tasks_easy[10:20] + tasks_difficult[10:20]
    tasks = copy.deepcopy(tasks_easy[0:10] + tasks_difficult[0:10])
    testing_tasks = copy.deepcopy(tasks_easy[10:20] + tasks_difficult[10:20])

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

    with open('TaDeLL_model_k_2_temp.pkl', 'wb') as f:
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

def eval():
    # This is a function to run the performance of TaDeLL_Model of different modes (Easy, difficult, mix) on the same task
    niter = 50
    # tasks0 = []
    # rewards_pg = []
    # values = []

    # Step 1: Load TaDeLL models
    with open('TaDeLL_model_k_2_easy.pkl', 'rb') as f:
        means_pg, means_tadell, niter, TaDeLL_Model, tasks0, tasks, testing_tasks, testing_tasks_pg, testing_tasks_TaDeLL = pickle.load(f)
    TM_easy = TaDeLL_Model

    with open('TaDeLL_model_k_2_difficult.pkl', 'rb') as f:
        means_pg, means_tadell, niter, TaDeLL_Model, tasks0, tasks, testing_tasks, testing_tasks_pg, testing_tasks_TaDeLL = pickle.load(f)
    TM_difficult = TaDeLL_Model

    with open('TaDeLL_model_k_2_mix.pkl', 'rb') as f:
        means_pg, means_tadell, niter, TaDeLL_Model, tasks0, tasks, testing_tasks, testing_tasks_pg, testing_tasks_TaDeLL = pickle.load(f)
    TM_mix = TaDeLL_Model

    # Step 2: Load task
    # task = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
    #             cpu_max=random.randint(30, 50), p=0.4 * np.random.random_sample() + 0.3, d=2, k=2)
    with open('Tasks.pkl', 'rb') as f:
        tasks00 = pickle.load(f)  # The task.policy is already the optimal policy
    task = tasks00[1][34]
    task_pg = copy.deepcopy(task)
    task_easy = copy.deepcopy(task)
    task_difficult = copy.deepcopy(task)
    task_mix = copy.deepcopy(task)

    # Step 3: Learn with PG and TaDeLL
    print("start PG test learning")
    rewards_pg = pg_rl(task_pg, niter)

    print("start TaDeLL test learning")
    print("testing for TaDeLL easy:")
    TM_easy.getDictPolicy_Single(task_easy)  # get warm start policy for testing tasks
    rewards_TaDeLL_easy = pg_rl(task_easy, niter)
    print("testing for TaDeLL difficult:")
    TM_difficult.getDictPolicy_Single(task_difficult)  # get warm start policy for testing tasks
    rewards_TaDeLL_difficult = pg_rl(task_difficult, niter)
    print("testing for TaDeLL mix:")
    TM_mix.getDictPolicy_Single(task_mix)  # get warm start policy for testing tasks
    rewards_TaDeLL_mix = pg_rl(task_mix, niter)

    with open('TaDeLL_result_k_2_eval_Tasks[][]_temp.pkl', 'wb') as f:
        pickle.dump([rewards_pg, rewards_TaDeLL_easy,rewards_TaDeLL_difficult,rewards_TaDeLL_mix, niter, task, task_pg, task_easy, task_difficult, task_mix], f)


    # Step 4: Painting
    fig, ax = plt.subplots()
    ax.plot(np.arange(niter), rewards_pg, label='PG')
    ax.plot(np.arange(niter), rewards_TaDeLL_easy, label='TaDeLL Easy')
    ax.plot(np.arange(niter), rewards_TaDeLL_difficult, label='TaDeLL Difficult')
    ax.plot(np.arange(niter), rewards_TaDeLL_mix, label='TaDeLL Mix')
    ax.legend()  # Add a legend.
    ax.set_xlabel('Iteration')  # Add an x-label to the axes.
    ax.set_ylabel('Averaged Reward')  # Add a y-label to the axes.
    ax.set_title("Comparison between PG and TaDeLL")  # Add a title to the axes.
    # fig.show()
    # plt.ioff()
    print("Hello Baby")

if __name__ == '__main__':
    # main()
    # test_feature_function()
    # pg_task_generation(30)
    eval()
    d = 1