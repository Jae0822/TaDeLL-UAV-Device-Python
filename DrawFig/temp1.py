import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import pickle
import matplotlib.pyplot as plt
import math
from statistics import mean
from collections import namedtuple
import copy

# from main import painting


with open('fig_D30.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)




def painting(avg):
    fig0, ax0 = plt.subplots(1)
    [plt.scatter(D.location[0], D.location[1]) for D in env.Devices]
    x = [D.location[0] for D in env.Devices]
    y = [D.location[1] for D in env.Devices]
    No = list(range(len(env.Devices)))
    # ax.scatter(x, y)
    for i, txt in enumerate(No):
        ax0.annotate(txt, (x[i], y[i]))
    ax0.plot([0],[0], label = 'V_Lim:' + str(param['V_Lim']) + ',  V:' + str(param['V']))
    ax0.set_xlim(0, 1000)
    ax0.set_ylim(0, 1000)
    ax0.legend(loc="best")
    ax0.grid(True)


    # †††††††††††††††††††††††††††††††††††††††Plotting Phase††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    fig, ax = plt.subplots(1)
    # ax[0].plot(np.arange(i_episode), Ep_reward, label='Actor-Critic')
    # ax[0].set_xlabel('Episodes')  # Add an x-label to the axes.
    # ax[0].set_ylabel('ep_reward')  # Add a y-label to the axes.
    # ax[0].set_title("The ep_reward")  # Add a title to the axes.

    # ax.plot(np.arange(1, EP+1), Ave_Reward, label='%.0f  Devices, %.0f TimeUnits, %.0f  episodes' %(param['num_Devices'], param['nTimeUnits'], EP, args.gamma, args.learning_rate))
    ax.plot(np.arange(1, param['episodes'] + 1), avg['Ave_Reward'],
            label=str(param['num_Devices']) + ' Devices,' + str(param['episodes']) + ' episodes,' + str(
                param['nTimeUnits']) + ' TimeUnits,' + str(param['gamma']) + ' gamma,' + str(
                param['learning_rate']) + ' lr,' + str(param['alpha']) + ' alpha, ' + str(param['mu']) + ' mu')
    ax.set_xlabel('Episodes')  # Add an x-label to the axes.
    ax.set_ylabel('Ave_Reward')  # Add a y-label to the axes.
    ax.set_title("The Ave_Reward, NN:" + str(model.pattern))  # Add a title to the axes.
    ax.axhline(y=max(avg['Ave_Reward']), color='r', linestyle='--', linewidth='0.9', label='Smart: ' + str(max(avg['Ave_Reward'])))
    ax.axhline(y=avg['ave_Reward_random'] * len(env_random.UAV.Reward), color='b', linestyle='--', linewidth='0.9',
               label='Random:' + str(avg['ave_Reward_random']*len(env_random.UAV.Reward)))
    ax.axhline(y=avg['ave_Reward_force'] * len(env_force.UAV.Reward), color='g', linestyle='--', linewidth='0.9', label='Forced:' + str(avg['ave_Reward_force']* len(env_force.UAV.Reward)))
    ax.legend(loc="best")

    # ax[1].plot(np.arange(i_episode), [ave_Reward_random]*i_episode, label='Random')
    # ax[1].set_xlabel('Episodes')  # Add an x-label to the axes.
    # ax[1].set_ylabel('Ave_Reward')  # Add a y-label to the axes.
    # ax[1].set_title("The Ave_Reward")  # Add a title to the axes.
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # †††††††††††††††††††††††††††††††††††††††EP_REWARD††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    fig_ep, ax_ep = plt.subplots(1)
    # ax.plot(np.arange(1, EP+1), Ave_Reward, label='%.0f  Devices, %.0f TimeUnits, %.0f  episodes' %(param['num_Devices'], param['nTimeUnits'], EP, args.gamma, args.learning_rate))
    ax_ep.plot(np.arange(1, param['episodes'] + 1), avg['Ep_reward'],
               label=str(param['num_Devices']) + ' Devices,' + str(param['episodes']) + ' episodes,' + str(
                   param['nTimeUnits']) + ' TimeUnits,' + str(param['gamma']) + ' gamma,' + str(
                   param['learning_rate']) + ' lr,' + str(param['alpha']) + ' alpha, ' + str(param['mu']) + ' mu')
    ax_ep.set_xlabel('Episodes')  # Add an x-label to the axes.
    ax_ep.set_ylabel('Ave_Reward')  # Add a y-label to the axes.
    ax_ep.set_title("The reward divided by number of flights, NN:" + str(model.pattern))  # Add a title to the axes.
    # ax.axhline(y=max(avg['Ave_Reward']), color='r', linestyle='--', linewidth='0.9',
    #            label='Smart: ' + str(max(avg['Ave_Reward'])))
    ax_ep.axhline(y=avg['ave_Reward_random'], color='b', linestyle='--', linewidth='0.9',
                  label='Random:' + str(avg['ave_Reward_random']))
    ax_ep.axhline(y=avg['ave_Reward_force'], color='g', linestyle='--', linewidth='0.9',
                  label='Forced:' + str(avg['ave_Reward_force']))
    ax_ep.legend(loc="best")

    # EP_reward复现SUMMATION
    # a = []
    # for x in range(1, param['episodes'] + 1):
    #     # print(x)
    #     a.append(len(logging_timeline[0][x]['UAV_Reward']))
    # fig_re, ax_re = plt.subplots(1)
    # # ax.plot(np.arange(1, EP+1), Ave_Reward, label='%.0f  Devices, %.0f TimeUnits, %.0f  episodes' %(param['num_Devices'], param['nTimeUnits'], EP, args.gamma, args.learning_rate))
    # ax_re.plot(np.arange(1, param['episodes'] + 1), [x * y for x, y in zip(a, avg['Ep_reward'])],
    #         label=str(param['num_Devices']) + ' Devices,' + str(param['episodes']) + ' episodes,' + str(
    #             param['nTimeUnits']) + ' TimeUnits,' + str(param['gamma']) + ' gamma,' + str(
    #             param['learning_rate']) + ' lr,' + str(param['alpha']) + ' alpha, ' + str(param['mu']) + ' mu')
    # ax_re.set_xlabel('Episodes')  # Add an x-label to the axes.
    # ax_re.set_ylabel('Ave_Reward')  # Add a y-label to the axes.
    # ax_re.set_title("The Ave_Reward, NN:" + str(model.pattern))  # Add a title to the axes.
    # # ax.axhline(y=max(avg['Ave_Reward']), color='r', linestyle='--', linewidth='0.9',
    # #            label='Smart: ' + str(max(avg['Ave_Reward'])))
    # ax_re.axhline(y=avg['ave_Reward_random'] * len(env_random.UAV.Reward), color='b', linestyle='--', linewidth='0.9',
    #            label='Random:' + str(avg['ave_Reward_random'] * len(env_random.UAV.Reward)))
    # ax_re.axhline(y=avg['ave_Reward_force'] * len(env_force.UAV.Reward), color='g', linestyle='--', linewidth='0.9',
    #            label='Forced:' + str(avg['ave_Reward_force'] * len(env_force.UAV.Reward)))
    # ax_re.legend(loc="best")

    # †††††††††††††††††††††††††††††††††††††††Smart††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    x = param['episodes']
    fig1, ax1 = plt.subplots(param['num_Devices'])
    fig1.supxlabel('Time Unites for one episode')
    fig1.supylabel('The Ave Reward')
    fig1.suptitle('The episode %.0f' % (x))
    for i in range(param['num_Devices']):
        if len(logging_timeline[i][x]['intervals']) == 0:
            ax1[i].text(0.4, 0.5, 'No Visit by UAV')
        else:
            ax1[i].step(logging_timeline[i][x]['KeyTime'], logging_timeline[i][x]['KeyRewards'], '^-g', where='post',
                        label='device %.0f' % (i))
            ax1[i].set_xlim([0, param['nTimeUnits']])
            ax1[i].axhline(y=logging_timeline[i][x]['avg_reward'], color='r', linestyle='--', linewidth='0.9',
                           label=logging_timeline[i][x]['avg_reward'])
            ax1[i].legend(loc="best")
            # ax1[i].text(2, logging_timeline[i][x]['avg_reward'], logging_timeline[i][x]['avg_reward'], verticalalignment='bottom',horizontalalignment='left', rotation=360, color='r')
            ax1[i].set_ylabel('device  %.0f' % (i))
            # if i == 0:
            #     ax1[i].set_title(model.pattern)
            # ax1[i].set_title('CPU Capacity: %.0f' % (env.Devices[i].cpu_capacity))
            for vv in range(len(np.where(logging_timeline[i][x]['TimeList'])[0])):
                ax1[i].axvline(x=np.where(logging_timeline[i][x]['TimeList'])[0][vv], linestyle='--', linewidth='0.9')
                # ax1[i].plot([np.where(Devices[i].TimeList)],[logging_timeline[i][x]['rewards']], 'o')
    # plt.show()
    #      https://matplotlib.org/stable/tutorials/text/text_intro.html

    # †††††††††††††††††††††††††††††††††††††††RANDOM††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    fig3, ax3 = plt.subplots(param['num_Devices'])
    fig3.supxlabel('Time Unites for one episode')
    fig3.supylabel('The Ave Reward')
    fig3.suptitle('The Random')
    for i in range(param['num_Devices']):
        if len(logging_timeline[i][0]['Random_intervals']) == 0:
            ax3[i].text(0.4, 0.5, 'No Visit by UAV')
        else:
            ax3[i].step(logging_timeline[i][0]['Random_KeyTime'], logging_timeline[i][0]['Random_KeyRewards'], '^-g',
                        where='post',
                        label='device %.0f' % (i))
            ax3[i].set_xlim([0, param['nTimeUnits']])
            ax3[i].axhline(y=logging_timeline[i][0]['Random_avg_reward'], color='r', linestyle='--', linewidth='0.9',
                           label=logging_timeline[i][0]['Random_avg_reward'])
            ax3[i].legend(loc="best")
            # ax1[i].text(2, logging_timeline[i][x]['avg_reward'], logging_timeline[i][x]['avg_reward'], verticalalignment='bottom',horizontalalignment='left', rotation=360, color='r')
            ax3[i].set_ylabel('device  %.0f' % (i))
            # if i == 0:
            #     ax1[i].set_title(model.pattern)
            ax3[i].set_title('CPU Capacity: %.0f' % (env_random.Devices[i].cpu_capacity))
            for vv in range(len(np.where(logging_timeline[i][0]['Random_TimeList'])[0])):
                ax3[i].axvline(x=np.where(logging_timeline[i][0]['Random_TimeList'])[0][vv], linestyle='--',
                               linewidth='0.9')
                # ax1[i].plot([np.where(Devices[i].TimeList)],[logging_timeline[i][x]['rewards']], 'o')


    # †††††††††††††††††††††††††††††††††††††††Force††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    fig4, ax4 = plt.subplots(param['num_Devices'])
    fig4.supxlabel('Time Unites for one episode')
    fig4.supylabel('The Ave Reward')
    fig4.suptitle('The Force')
    for i in range(param['num_Devices']):
        if len(logging_timeline[i][0]['Force_intervals']) == 0:
            ax4[i].text(0.4, 0.5, 'No Visit by UAV')
        else:
            ax4[i].step(logging_timeline[i][0]['Force_KeyTime'], logging_timeline[i][0]['Force_KeyRewards'], '^-g',
                        where='post',
                        label='device %.0f' % (i))
            ax4[i].set_xlim([0, param['nTimeUnits']])
            ax4[i].axhline(y=logging_timeline[i][0]['Force_avg_reward'], color='r', linestyle='--', linewidth='0.9',
                           label=logging_timeline[i][0]['Force_avg_reward'])
            ax4[i].legend(loc="best")
            # ax1[i].text(2, logging_timeline[i][x]['avg_reward'], logging_timeline[i][x]['avg_reward'], verticalalignment='bottom',horizontalalignment='left', rotation=360, color='r')
            ax4[i].set_ylabel('device  %.0f' % (i))
            # if i == 0:
            #     ax1[i].set_title(model.pattern)
            ax4[i].set_title('CPU Capacity: %.0f' % (env_force.Devices[i].cpu_capacity))
            for vv in range(len(np.where(logging_timeline[i][0]['Force_TimeList'])[0])):
                # ax1[i].plot([np.where(Devices[i].TimeList)],[logging_timeline[i][x]['rewards']], 'o')
                ax4[i].axvline(x=np.where(logging_timeline[i][0]['Force_TimeList'])[0][vv], linestyle='--',
                               linewidth='0.9')


    # fig2, ax2 = plt.subplots(1)
    # type = ['Random', 'Force', 'Smart']
    # # data1 = [np.mean([i for i in Reward_random if i >= -30]), np.mean([i for i in Reward_force if i >= -30]), np.mean(logging_timeline[0][param['episodes']]['UAV_Reward'])]
    # # data2 = [np.mean(PV_random), np.mean(PV_force), np.mean(logging_timeline[0][param['episodes']]['UAV_Energy'])]
    # # ax2.bar(type, data1, label = 'reward')
    # # ax2.bar(type, data2, bottom=np.array(data1), label = 'energy')
    # # data11 = [- np.mean([i for i in env_random.UAV.Reward if i >= -30]),
    # #           -np.mean([i for i in env_force.UAV.Reward if i >= -30]),
    # #           -np.mean(logging_timeline[0][param['episodes']]['UAV_Reward'])]
    # data111 = [-np.mean(env_random.UAV.Reward), -np.mean(env_force.UAV.Reward),
    #            -np.mean(logging_timeline[0][param['episodes']]['UAV_Reward'])]
    # data1111 = [-np.sum(env_random.UAV.Reward), -np.sum(env_force.UAV.Reward),
    #             -np.sum(logging_timeline[0][param['episodes']]['UAV_Reward'])]
    # data22 = [np.mean(env_random.UAV.Energy), np.mean(env_force.UAV.Energy),
    #           np.mean(logging_timeline[0][param['episodes']]['UAV_Energy'])]
    # data222 = [np.sum(env_random.UAV.Energy), np.sum(env_force.UAV.Energy),
    #            np.sum(logging_timeline[0][param['episodes']]['UAV_Energy'])]
    # # [a + b for (a, b) in zip(data111, data22)]
    # # [a + b for (a, b) in zip(data1111, data222)]
    # ax2.bar(type, [k * param['mu'] for k in data1111], label='reward')
    # ax2.bar(type, [k * param['mu'] for k in data222], bottom=np.array(data1111) * param['mu'], label='energy')
    # ax2.axhline(y=0, color='k', linestyle='-', linewidth='0.6')
    # ax2.legend(loc="best")
    # fig2.suptitle('The Sum')
    # # ax2.set_xlabel('Different Types')  # Add an x-label to the axes.
    # ax2.set_ylabel('Total Cost')  # Add a y-label to the axes.
    # plt.show()
    # https: // www.zhihu.com / question / 507977476 / answer / 2283372858  (画叠加柱状图)
    # https: // juejin.cn / post / 6844903664780328974

    # fig5, ax5 = plt.subplots(1)
    # type = ['Random', 'Force', 'Smart']
    # # data1 = [np.mean([i for i in Reward_random if i >= -30]), np.mean([i for i in Reward_force if i >= -30]), np.mean(logging_timeline[0][param['episodes']]['UAV_Reward'])]
    # # data2 = [np.mean(PV_random), np.mean(PV_force), np.mean(logging_timeline[0][param['episodes']]['UAV_Energy'])]
    # # ax2.bar(type, data1, label = 'reward')
    # # ax2.bar(type, data2, bottom=np.array(data1), label = 'energy')
    # # data11 = [- np.mean([i for i in env_random.UAV.Reward if i >= -30]),
    # #           -np.mean([i for i in env_force.UAV.Reward if i >= -30]),
    # #           -np.mean(logging_timeline[0][param['episodes']]['UAV_Reward'])]
    # data111 = [-np.mean(env_random.UAV.Reward), -np.mean(env_force.UAV.Reward),
    #            -np.mean(logging_timeline[0][param['episodes']]['UAV_Reward'])]
    # data1111 = [-np.sum(env_random.UAV.Reward), -np.sum(env_force.UAV.Reward),
    #             -np.sum(logging_timeline[0][param['episodes']]['UAV_Reward'])]
    # data22 = [np.mean(env_random.UAV.Energy), np.mean(env_force.UAV.Energy),
    #           np.mean(logging_timeline[0][param['episodes']]['UAV_Energy'])]
    # data222 = [np.sum(env_random.UAV.Energy), np.sum(env_force.UAV.Energy),
    #            np.sum(logging_timeline[0][param['episodes']]['UAV_Energy'])]
    # ax5.bar(type, [k * param['mu'] for k in data111], label='reward')
    # ax5.bar(type, [k * param['mu'] for k in data22], bottom=np.array(data111) * param['mu'], label='energy')
    # ax5.axhline(y=0, color='k', linestyle='-', linewidth='0.6')
    # ax5.legend(loc="best")
    # fig5.suptitle('The Mean')
    # # ax2.set_xlabel('Different Types')  # Add an x-label to the axes.
    # ax5.set_ylabel('Total Cost')  # Add a y-label to the axes.
    # plt.show()


    d = 1
    # †††††††††††††††††††††††††††††††††††††††柱状图††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    x = param['episodes']
    fig7, ax7 = plt.subplots(2, sharex=True)
    fig7.suptitle('Devcie and UAV cost')
    type = ['Random', 'Force', 'Smart']
    data111 = [-np.mean(env_random.UAV.Reward), -np.mean(env_force.UAV.Reward),
               -np.mean(logging_timeline[0][x]['UAV_Reward'])]
    data1111 = [-np.sum(env_random.UAV.Reward), -np.sum(env_force.UAV.Reward),
                -np.sum(logging_timeline[0][x]['UAV_Reward'])]
    data22 = [np.mean(env_random.UAV.Energy), np.mean(env_force.UAV.Energy),
              np.mean(logging_timeline[0][x]['UAV_Energy'])]
    data222 = [np.sum(env_random.UAV.Energy), np.sum(env_force.UAV.Energy),
               np.sum(logging_timeline[0][x]['UAV_Energy'])]
    ax7[0].bar(type, [k * param['mu'] for k in data111], label='reward')
    ax7[0].bar(type, [k * param['mu'] for k in data22], bottom=np.array(data111) * param['mu'], label='energy')
    ax7[0].axhline(y=0, color='k', linestyle='-', linewidth='0.6')
    ax7[0].legend(loc="best")
    ax7[0].set_ylabel('Total Cost')  # Add a y-label to the axes.
    ax7[0].set_title('The Mean')
    ax7[1].bar(type, [k * param['mu'] for k in data1111], label='reward')
    ax7[1].bar(type, [k * param['mu'] for k in data222], bottom=np.array(data1111) * param['mu'], label='energy')
    ax7[1].axhline(y=0, color='k', linestyle='-', linewidth='0.6')
    ax7[1].legend(loc="best")
    ax7[1].set_ylabel('Total Cost')  # Add a y-label to the axes.
    ax7[1].set_title('The Sum')


    # fig6, ax6 = plt.subplots(1)
    # type = ['Random', 'Force', 'Smart']
    # dataAoImean = [np.mean(env_random.UAV.AoI), np.mean(env_force.UAV.AoI),
    #            np.mean(logging_timeline[0][param['episodes']]['UAV_AoI'])]
    # dataAoIsum = [np.sum(env_random.UAV.AoI), np.sum(env_force.UAV.AoI),
    #             np.sum(logging_timeline[0][param['episodes']]['UAV_AoI'])]
    # dataCPUmean = [np.mean(env_random.UAV.CPU), np.mean(env_force.UAV.CPU),
    #           np.mean(logging_timeline[0][param['episodes']]['UAV_CPU'])]
    # dataCPUsum = [np.sum(env_random.UAV.CPU), np.sum(env_force.UAV.CPU),
    #            np.sum(logging_timeline[0][param['episodes']]['UAV_CPU'])]
    # ax6.bar(type, [k * param['beta'] for k in dataAoImean], label='AoI')
    # ax6.bar(type, [k * param['beta'] for k in dataCPUmean], bottom=np.array(dataAoImean) * param['beta'], label='CPU')
    # ax6.axhline(y=0, color='k', linestyle='-', linewidth='0.6')
    # ax6.legend(loc="best")
    # fig6.suptitle('The Mean')
    # ax6.set_ylabel('Total Cost')  # Add a y-label to the axes.
    # plt.show()

    # fig7, ax7 = plt.subplots(1)
    # type = ['Random', 'Force', 'Smart']
    # dataAoImean = [np.mean(env_random.UAV.AoI), np.mean(env_force.UAV.AoI),
    #            np.mean(logging_timeline[0][param['episodes']]['UAV_AoI'])]
    # dataAoIsum = [np.sum(env_random.UAV.AoI), np.sum(env_force.UAV.AoI),
    #             np.sum(logging_timeline[0][param['episodes']]['UAV_AoI'])]
    # dataCPUmean = [np.mean(env_random.UAV.CPU), np.mean(env_force.UAV.CPU),
    #           np.mean(logging_timeline[0][param['episodes']]['UAV_CPU'])]
    # dataCPUsum = [np.sum(env_random.UAV.CPU), np.sum(env_force.UAV.CPU),
    #            np.sum(logging_timeline[0][param['episodes']]['UAV_CPU'])]
    # ax7.bar(type, [k * param['beta'] for k in dataAoIsum], label='AoI')
    # ax7.bar(type, [k * param['beta'] for k in dataCPUsum], bottom=np.array(dataAoIsum) * param['beta'], label='CPU')
    # ax7.axhline(y=0, color='k', linestyle='-', linewidth='0.6')
    # ax7.legend(loc="best")
    # fig7.suptitle('The Sum')
    # ax7.set_ylabel('Total Cost')  # Add a y-label to the axes.
    # plt.show()
    x = param['episodes']
    fig8, ax8 = plt.subplots(2, sharex=True)
    fig8.suptitle('AoI and CPU cost')
    type = ['Random', 'Force', 'Smart']
    dataAoImean = [np.mean(env_random.UAV.AoI), np.mean(env_force.UAV.AoI),
               np.mean(logging_timeline[0][x]['UAV_AoI'])]
    dataAoIsum = [np.sum(env_random.UAV.AoI), np.sum(env_force.UAV.AoI),
                np.sum(logging_timeline[0][x]['UAV_AoI'])]
    dataCPUmean = [np.mean(env_random.UAV.CPU), np.mean(env_force.UAV.CPU),
              np.mean(logging_timeline[0][x]['UAV_CPU'])]
    dataCPUsum = [np.sum(env_random.UAV.CPU), np.sum(env_force.UAV.CPU),
               np.sum(logging_timeline[0][x]['UAV_CPU'])]
    ax8[0].bar(type, [k * param['beta'] for k in dataAoImean], label='AoI')
    ax8[0].bar(type, [k * param['beta'] for k in dataCPUmean], bottom=np.array(dataAoImean) * param['beta'], label='CPU')
    ax8[0].axhline(y=0, color='k', linestyle='-', linewidth='0.6')
    ax8[0].legend(loc="best")
    ax8[0].set_title('The Mean')
    ax8[0].set_ylabel('Total Cost')  # Add a y-label to the axes.
    ax8[1].bar(type, [k * param['beta'] for k in dataAoIsum], label='AoI')
    ax8[1].bar(type, [k * param['beta'] for k in dataCPUsum], bottom=np.array(dataAoIsum) * param['beta'], label='CPU')
    ax8[1].axhline(y=0, color='k', linestyle='-', linewidth='0.6')
    ax8[1].legend(loc="best")
    ax8[1].set_title('The Sum')
    ax8[1].set_ylabel('Total Cost')  # Add a y-label to the axes.

    # †††††††††††††††††††††††††††††††††††††††速度图††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    V_avg = []
    for x in range(1, param['episodes']):
        V_avg.append(mean(logging_timeline[0][x]['UAV_VelocityList']))
    fig_v, ax_v = plt.subplots(1)
    ax_v.plot(np.arange(1, param['episodes']), V_avg)
    ax_v.set_ylabel('Velocity(m/s)')
    ax_v.set_xlabel('episodes')
    ax_v.grid(True)
    fig_v.suptitle('Velocity trend')
    plt.show()




painting(avg)



luck = 1