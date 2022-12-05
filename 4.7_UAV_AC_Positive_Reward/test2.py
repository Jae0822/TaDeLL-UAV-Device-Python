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



class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, 32)
        self.affine2 = nn.Linear(32, 64)
        self.pattern = [32, 64]

        # actor's layer
        self.action_head = nn.Linear(64, output_size)
        # critic's layer
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_prob, state_values








    for (log_prob, value, velocity), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))















