import numpy as np
import random
import pickle

import torch

from NNStrategy import NNStrategy
from RandomStrategy import RandomStrategy
from ForcedStrategy import ForcedStrategy
import Util


def main():

    # Prepare the environment and devices
    #  V: 72 km/h =  20 m/s
    #  field: 1 km * 1km
    #  dist:
    length = 2000
    param = {'episodes': 25, 'nTimeUnits': length, 'nTimeUnits_random': length, 'nTimeUnits_force': length,
             'gamma': 0.99, 'learning_rate': 0.07, 'log_interval': 1, 'seed': 0, 'alpha': 2, 'mu': 0.5, 'beta': 0.5,
             'num_Devices': 25, 'V': 25, 'V_Lim': 40, 'field': 1000, 'dist': 0.040, 'freq_low': 8, 'freq_high': 16,
             'cpu_capacity' : 50}

    random.seed(param['seed'])
    np.random.seed(param['seed'])
    torch.manual_seed(param['seed'])
    torch.use_deterministic_algorithms(True)
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
                {'intervals': [], 'rewards': [], 'timeline': [], 'plt_reward': [], 'avg_reward': []})

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
    avg['Ave_Reward'] = 0
    avg['Ep_reward'] = 0
    avg['ave_Reward_random'] = random_strategy.ave_Reward_random
    avg['ave_Reward_force'] = forced_strategy.ave_Reward_force
    # with open('fig_temp.pkl', 'wb') as f:
    #     pickle.dump([model, env, param, avg, logging_timeline], f)

    with open('fig_temp.pkl', 'wb') as f:
        pickle.dump([nn_strategy.model, nn_strategy.env, random_strategy.env, forced_strategy.env, param, avg, logging_timeline], f)

    # †††††††††††††††††††††††††††††††††††††††Painting††††††††††††††††††††††††††††††††††††††††††††††††††††††††††
    Util.painting(avg, param, nn_strategy.env, nn_strategy.model, random_strategy.env, forced_strategy.env, logging_timeline)
    # †††††††††††††††††††††††††††††††††††††††Painting††††††††††††††††††††††††††††††††††††††††††††††††††††††††††



if __name__ == '__main__':
    main()