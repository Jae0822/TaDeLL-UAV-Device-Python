"""
用SEABORN画速度收敛图的尝试
"""

import numpy as np
import pandas as pd
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so


with open('fig_A19.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)

df = pd.DataFrame()
for x in range(1, param['episodes']):
    # temp = logging_timeline[0][x]['UAV_VelocityList']
    v = logging_timeline[0][x]['UAV_VelocityList']
    dd = {'episode': x, 'steps': np.arange(1, len(v) + 1), 'v': v}
    df0 = pd.DataFrame(data=dd)
    df = df.append(df0)


sns.lineplot(data=df, x="episode", y="v")


d = 1