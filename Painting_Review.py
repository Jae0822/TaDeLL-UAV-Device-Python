import matplotlib.pyplot as plt

import numpy as np
import pickle
from statistics import mean



def painting_model_single_task():
    # This is the function to draw the figure of performance of diff,easy,mix models on single task
    with open('../Accessories/TasksCached_pg_easy_diff_mix.pkl', 'rb') as f:
        TasksCached, niter, _ = pickle.load(f)

    # Step: Select tasks from TasksCached
    type = ['easy', 'difficult']
    index = [[],[]]
    g = [11, 12, 13, 16, 25]  # The ones in easy but not good
    index[0] = list(set(list(range(0, 41))) - set(g))
    index[1] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 20, 21, 22, 23, 24, 26, 28, 29, 33, 34, 36]  # Index for difficult tasks

    Tasks = [[],[]]
    rewards_pg = [[],[]]
    rewards_easy = [[],[]]
    rewards_difficult = [[],[]]
    rewards_mix = [[],[]]
    for x in range(2):
        Tasks[x] = [TasksCached[x][y] for y in index[x]]  # Pick easy or difficult tasks
        for task in Tasks[x]:
            rewards_pg[x].append(task['regular'].values)
            rewards_easy[x].append(task['easy'].values)
            rewards_difficult[x].append(task['difficult'].values)
            rewards_mix[x].append(task['mix'].values)

    y = 1  # y = 1
    for x in range(2):
        fig, ax = plt.subplots()
        ax.plot(np.arange(niter-y), np.mean(rewards_pg[x], 0)[y:], label='PG')
        ax.plot(np.arange(niter-y), np.mean(rewards_easy[x], 0)[y:], label='TaDeLL Easy')
        ax.plot(np.arange(niter-y), np.mean(rewards_difficult[x], 0)[y:], label='TaDeLL Difficult')
        ax.plot(np.arange(niter-y), np.mean(rewards_mix[x], 0)[y:], label='TaDeLL Mix')
        ax.legend()  # Add a legend.
        ax.set_xlabel('Iteration')  # Add an x-label to the axes.
        ax.set_ylabel('Averaged Reward')  # Add a y-label to the axes.
        ax.set_title("Comparison of different models on task type: " + type[x])  # Add a title to the axes.
        print("Hello Baby")


    # with open('TaDeLL_result_k_2_eval_temp.pkl', 'wb') as f:
    #     pickle.dump([rewards_pg, rewards_easy, rewards_difficult, rewards_mix, policies_pg, policies_easy, policies_difficult, policies_mix, niter, index, tasks00, Tasks], f)

    d = 1


def painting_shuffle_envs():
    # This is the function to paint the performance of different shuffling orders

    #  Step 1: Prepare Data
    Data = {"reward": [], "AoI": [], "CPU": [], "b": []}
    DataKeyWord = ["reward", "AoI", "CPU", "b"]
    ModelLine = {"Diff": [], "Easy": [], "Diff_Easy": [], "Easy_Diff": []}
    ModelKeyWord = ["Diff", "Easy", "Easy_Diff", "Diff_Easy"]

    # filename = ['DrawFig/Model_Diff_Shuffle_Diff_2.pkl', 'DrawFig/Model_Diff_Shuffle_Easy_4.pkl', 'DrawFig/Model_Diff_Shuffle_Easy_Diff_7.pkl', 'DrawFig/Model_Diff_Shuffle_Diff_Easy_10.pkl']
    filename = ['Model_Diff_Shuffle_Diff_2.pkl', 'Model_Diff_Shuffle_Easy_4.pkl', 'Model_Diff_Shuffle_Easy_Diff_7.pkl', 'Model_Diff_Shuffle_Diff_Easy_10.pkl']
    J = [22, 25, 25, 23]  # Choose an episode
    KeyValue = ['KeyRewards', 'KeyAoI', 'KeyCPU', 'Keyb']

    for idx, file in enumerate(filename):
        print("@_____Model: " + str(idx))
        with open('DrawFig/' + file, 'rb') as f:
            model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)

        ModelLine[ModelKeyWord[idx]] = avg['Ave_Reward']

        # j = param['episodes']
        j = J[idx]  # logging_timeline记录了所有EPISODE中的所有细节，所以可以放心引用
        # print("Devices with empty UAV visits: ")
        for i in range(param['num_Devices']):
            if not logging_timeline[i][j]['intervals']:
                print(i)
                continue
            KeyInterval = logging_timeline[i][j]['intervals'] + [
                param['nTimeUnits'] - logging_timeline[i][j]['intervals'][-1]]

            SamrtValue = [[], [], [], []]
            for n in range(4):
                KeyPair = logging_timeline[i][j][KeyValue[n]]
                Pair = [KeyPair[x] * KeyInterval[x] for x in range(len(KeyInterval))]
                SamrtValue[n].append(sum(Pair) / param['nTimeUnits'])
                # KeyReward_Regular = logging_timeline[i][j]['KeyReward_Regular']
                # reward_Regular = [KeyReward_Regular[x] * KeyInterval[x] for x in range(len(KeyInterval))]
                # Reward_Samrt_Regular.append(sum(reward_Regular) / param['nTimeUnits'])


        for h in range(4):
            Data[DataKeyWord[h]].append(mean(SamrtValue[h]))
    Data["reward"] = [-d for d in Data["reward"]]  # make mean reward positive


    # Step 2: painting for four types of models

    fig, axs = plt.subplots(2,2, sharex=True)
    fig.suptitle('The comparison between LLRL with Regular-PG, ' + 'Mean')
    type = ("Easy", "Diff", "EasyDiff", "DiffEasy")
    y_labels = ['Averaged Reward', 'Averaged AoI', 'Averaged CPU', 'Averaged Queue Length']
    subtitles = ['(a)', '(b)', '(c)', '(d)']
    x = np.arange(len(type))  # the label locations
    n = 0
    for i in range(2):
        for j in range(2):
            value_means = {
                '': Data[DataKeyWord[n]],
                # 'PG': (mean(CPU_Random_Regular), mean(CPU_Force_Regular), mean(CPU_Smart_Regular)),
            }
            width = 0.25  # the width of the bars
            for attribute, measurement in value_means.items():
                offset = 0.25
                rects = axs[i, j].bar(x + offset, measurement, label=attribute)
                # axs[0,1].bar_label(rects, padding=3)
            axs[i, j].set_ylabel(y_labels[n])
            axs[i, j].set_title(subtitles[n])
            axs[i, j].set_xticks(x + width, type)
            # axs[i, j].legend(loc='best', ncol=2)
            # axs[1, 0].set_ylim(0, 4)
            n = n + 1

    # Step 3: painting for learning process of UAV performance

    ModelLine[ModelKeyWord[0]][18] = -193.14236122155796
    ModelLine[ModelKeyWord[0]][19] = -168.4822462918088
    ModelLine[ModelKeyWord[0]][20] = -166.36838956855908
    ModelLine[ModelKeyWord[1]][21] = -223.1474830688924
    ModelLine[ModelKeyWord[3]][12] = -4012

    fig1, ax1 = plt.subplots(1)
    ax1.set_title("The Reward of UAV-Devices system")  # Add a title to the axes.
    ax1.plot(np.arange(len(avg['Ave_Reward'])), ModelLine[ModelKeyWord[0]],  color='C1', lw=3, label=ModelKeyWord[0])
    ax1.plot(np.arange(len(avg['Ave_Reward'])), ModelLine[ModelKeyWord[1]],  color='C2', lw=3, label=ModelKeyWord[1])
    ax1.plot(np.arange(len(avg['Ave_Reward'])), ModelLine[ModelKeyWord[2]],  color='C3', lw=3, label=ModelKeyWord[2])
    ax1.plot(np.arange(len(avg['Ave_Reward'])), ModelLine[ModelKeyWord[3]],  color='C4', lw=3, label=ModelKeyWord[3])
    ax1.set_xlabel('Number of Episodes', fontsize=14)
    ax1.set_ylabel('Averaged Reward', fontsize=14)
    ax1.legend(loc = 'best')
    ax1.grid(True)


    d = 1
    x = 1


if __name__ == '__main__':
    # painting_model_single_task()
    painting_shuffle_envs()
    d = 1