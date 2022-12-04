import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d
import math
import random

from env import Task

fig, ax = plt.subplots(1)
fig.suptitle('1')

fig2, ax2 = plt.subplots(1)
fig2.suptitle('2')

fig3, ax3 = plt.subplots(1)
fig3.suptitle('3')

fig4, ax4 = plt.subplots(1)
fig4.suptitle('4')

fig5, ax5 = plt.subplots(1)
fig5.suptitle('5')

fig6, ax6 = plt.subplots(1)
fig6.suptitle('6')

fig7, ax7 = plt.subplots(1)
fig7.suptitle('7')

fig8, ax8 = plt.subplots(2)
ax8[0].plot(range(10), 'r')
ax8[1].plot(range(10), 'r')
fig8.suptitle('8')

fig9, ax9 = plt.subplots(2,2)
ax9[0,0].plot(range(10), 'r')
ax9[0,1].plot(range(10), 'b')
ax9[1,0].plot(range(10), 'g')
ax9[1,1].plot(range(10), 'k')
fig9.suptitle('9')


tsk = Task(mean_packet_cycles=random.randint(15, 35), variance_packet_cycles=random.randint(3, 8),
            cpu_max=50, p=0.4 * np.random.random_sample() + 0.3, d=2, k=2)

values = tsk.get_value(tsk.policy['theta'])
aoi, cpu, b, rew = tsk.get_AoI_CPU(tsk.policy['theta'])

a = []

a.append(tsk.get_AoI_CPU(tsk.policy['theta']))
a.append(tsk.get_AoI_CPU(tsk.policy['theta']))
a.append(tsk.get_AoI_CPU(tsk.policy['theta']))
a.append(tsk.get_AoI_CPU(tsk.policy['theta']))
a.append(tsk.get_AoI_CPU(tsk.policy['theta']))
a.append(tsk.get_AoI_CPU(tsk.policy['theta']))

de = 1
