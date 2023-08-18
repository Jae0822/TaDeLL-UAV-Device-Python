import numpy as np
import copy
import matplotlib.pyplot as plt
from statistics import mean
import pickle
import math

import torch

from IoTEnv import Uav, Device, Env, Policy
from UAVEnergy import UAV_Energy
from NNStrategy import NNStrategy
from RandomStrategy import RandomStrategy
from ForcedStrategy import ForcedStrategy

def painting(avg, param, env, model, env_random, env_force, logging_timeline):
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
    # plt.show()

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



def main():

    # Prepare the environment and devices
    #  V: 72 km/h =  20 m/s
    #  field: 1 km * 1km
    #  dist:
    length = 2000
    param = {'episodes': 25, 'nTimeUnits': length, 'nTimeUnits_random': length, 'nTimeUnits_force': length,
             'gamma': 0.99, 'learning_rate': 0.07, 'log_interval': 1, 'seed': 0, 'alpha': 2, 'mu': 0.5, 'beta': 0.5,
             'num_Devices': 25, 'V': 25, 'V_Lim': 40, 'field': 1000, 'dist': 0.040, 'freq_low': 8, 'freq_high': 16}
    np.random.seed(param['seed'])
    torch.manual_seed(param['seed'])
    torch.set_num_interop_threads(4)
    torch.set_num_threads(4)

    # logging for each episode
    # logging_timeline = [ device0, device1, device2....,  ,  ]
    # device = [episode0, episode1, episode2, ...,  ]
    # episode = {'intervals': [], 'rewards': []}
    logging_timeline = []
    for i in range(param['num_Devices']):
        logging_timeline.append([])
        for j in range(param['episodes'] + 1):
            logging_timeline[i].append(
                {'intervals': [], 'rewards': [], 'rewards_regular': [], 'timeline': [], 'plt_reward': [], 'avg_reward': []})

    # log parameters
    print(param)

    # †††††††††††††††††††††††††††††††††††††††Smart Trajectory††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    nn_strategy = NNStrategy(param, logging_timeline)
    nn_strategy.learning()
    # †††††††††††††††††††††††††††††††††††††††Smart Trajectory††††††††††††††††††††††††††††††††††††††††††††††††††††††††††



    # †††††††††††††††††††††††††††††††††††††††Random Trajectory††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    random_strategy = RandomStrategy(param, logging_timeline)
    random_strategy.learning()
    # †††††††††††††††††††††††††††††††††††††††Random Trajectory††††††††††††††††††††††††††††††††††††††††††††††††††††††††††



    # †††††††††††††††††††††††††††††††††††††††Forced Trajectory††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    forced_strategy = ForcedStrategy(param, logging_timeline)
    forced_strategy.learning()
    # †††††††††††††††††††††††††††††††††††††††Forced Trajectory††††††††††††††††††††††††††††††††††††††††††††††††††††††††††



    avg = {}
    avg['Ave_Reward'] = nn_strategy.average_reward
    avg['Ep_reward'] = nn_strategy.episode_reward
    avg['ave_Reward_random'] = random_strategy.ave_Reward_random
    avg['ave_Reward_force'] = forced_strategy.ave_Reward_force
    # with open('fig_temp.pkl', 'wb') as f:
    #     pickle.dump([model, env, param, avg, logging_timeline], f)

    with open('fig_temp.pkl', 'wb') as f:
        pickle.dump([nn_strategy.model, nn_strategy.env, random_strategy.env, forced_strategy.env, param, avg, logging_timeline], f)

    # †††††††††††††††††††††††††††††††††††††††Painting††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    painting(avg, param, nn_strategy.env, nn_strategy.model, random_strategy.env, forced_strategy.env, logging_timeline)
    # †††††††††††††††††††††††††††††††††††††††Painting††††††††††††††††††††††††††††††††††††††††††††††††††††††††††



if __name__ == '__main__':
    main()