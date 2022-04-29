parser.add_argument('--nTimeUnits', type=int, default=300,
                    help='number of time units per episode')
parser.add_argument('--nDevices', type=int, default=3,
                    help='number of Devices')
parser.add_argument('--V', type=int, default=20,
                    help='the velocity of the UAV (unit: m/s)')
parser.add_argument('--dist', type=int, default=40,
                    help='The minimum distance between devices')
parser.add_argument('--field', type=int, default=200,
                    help='the edge length of the square field (unit: m)')
parser.add_argument('--interval', type=float, default=[200, 400, 600], nargs='+',
                    help='the new task interval of each device')

# Plotting procedure
plt.ion()
fig, ax = plt.subplots()
ax.plot(np.arange(niter), means_pg, label='PG')
ax.plot(np.arange(niter), means_pgella, label='PG-ELLA')
ax.legend()  # Add a legend.
ax.set_xlabel('Iteration')  # Add an x-label to the axes.
ax.set_ylabel('Averaged Reward')  # Add a y-label to the axes.
ax.set_title("Comparison between PG and PG-ELLA")  # Add a title to the axes.
fig.show()
plt.ioff()
print("Hello Baby")

# Plotting phase
plt.ion()
fig, ax = plt.subplots(2, num_Devices, sharex=True)
fig.suptitle('Learning Process')
for i in range(num_Devices):
    ax[0, i].plot(np.arange(nTimeUnits), Rewards_Random[i], label="Warm Start")
    ax[0, i].plot(np.arange(nTimeUnits), Rewards_Random_Natural[i], label="Natural")
    # ax[i].set_title('device')
    ax[0, i].legend()
plt.show()


# Plotting phase
ax[0].plot(np.arange(nTimeUnits), Rewards_Random[0], label='device 0')
ax[1].plot(np.arange(nTimeUnits), Rewards_Random[1], label='device 1')
# Replacement
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.plot()
ax1.set_title()
ax1.set_ylabel()
ax1.setylin()
plt.show()




















