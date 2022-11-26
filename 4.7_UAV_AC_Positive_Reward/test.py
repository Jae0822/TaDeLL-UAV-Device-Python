import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d

with open('fig_A6.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)




Devices = env.Devices


fig, ax = plt.subplots(1)
# for D in Devices:
#     plt.scatter(D.location[0], D.location[1])
[plt.scatter(D.location[0], D.location[1]) for D in Devices]
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.grid(True)