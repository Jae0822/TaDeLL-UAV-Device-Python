import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d

with open('fig_A2.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)


for i in range(param['num_Devices']):
    ls1 = [0] + logging_timeline[i][0]['Force_intervals']
    ls2 = logging_timeline[i][0]['Force_KeyRewards']
    if len(logging_timeline[i][0]['Force_KeyTime']) == 1:
        logging_timeline[i][0]['Force_avg_reward'] = None
    else:
        logging_timeline[i][0]['Force_avg_reward'] = sum([x * y for x, y in zip(ls1, ls2)]) / \
                                               logging_timeline[i][0]['Force_KeyTime'][-1]

for i in range(param['num_Devices']):
    ls1 = [0] + logging_timeline[i][0]['Random_intervals']
    ls2 = logging_timeline[i][0]['Random_KeyRewards']
    if len(logging_timeline[i][0]['Random_KeyTime']) == 1:
        logging_timeline[i][0]['Random_avg_reward'] = None
    else:
        logging_timeline[i][0]['Random_avg_reward'] = sum([x * y for x, y in zip(ls1, ls2)]) / \
                                               logging_timeline[i][0]['Random_KeyTime'][-1]