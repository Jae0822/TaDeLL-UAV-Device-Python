import numpy as np
import pickle
import matplotlib.pyplot as plt
from statistics import mean
import math

# fig_P02.pkl

with open('fig_P02.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)

x = 25
def CPU_J(cycles):
    # IoT device cycles into J
    J = pow(cycles * pow(10, 8), 3) * 4 * pow(10, -28)
    return J



#  Random
Reward_Random = []
AoI_Random = []
CPU_Random = []
b_Random = []
for i in range(param['num_Devices']):
    KeyInterval = env_random.Devices[i].intervals + [param['nTimeUnits_random'] - env_random.Devices[i].KeyTime[-1]]

    # Reward ( = AoI + CPU)
    KeyReward = logging_timeline[i][0]['Random_KeyRewards']
    reward = [KeyReward[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    Reward_Random.append(sum(reward) / param['nTimeUnits_random'])

    # AoI
    KeyAoI = logging_timeline[i][0]['Random_KeyAoI']
    AoI = [KeyAoI[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    AoI_Random.append(sum(AoI) / param['nTimeUnits_random'])

    # CPU
    KeyCPU = logging_timeline[i][0]['Random_KeyCPU']
    CPU = [KeyCPU[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    CPU_Random.append(sum(CPU) / param['nTimeUnits_random'])

    # b
    Keyb = logging_timeline[i][0]['Random_Keyb']
    b = [Keyb[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    b_Random.append(sum(b) / param['nTimeUnits_random'])


#  Force
Reward_Force = []
AoI_Force = []
CPU_Force = []
b_Force = []
for i in range(param['num_Devices']):
    KeyInterval = env_force.Devices[i].intervals + [param['nTimeUnits_force'] - env_force.Devices[i].KeyTime[-1]]

    # Reward
    KeyReward = logging_timeline[i][0]['Force_KeyRewards']
    reward = [KeyReward[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    Reward_Force.append(sum(reward) / param['nTimeUnits_force'])

    # AoI
    KeyAoI = logging_timeline[i][0]['Force_KeyAoI']
    AoI = [KeyAoI[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    AoI_Force.append(sum(AoI) / param['nTimeUnits_force'])

    # CPU
    KeyCPU = logging_timeline[i][0]['Force_KeyCPU']
    CPU = [KeyCPU[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    CPU_Force.append(sum(CPU) / param['nTimeUnits_force'])

    # b
    Keyb = logging_timeline[i][0]['Force_Keyb']
    b = [Keyb[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    b_Force.append(sum(b) / param['nTimeUnits_force'])


#  Smart
Reward_Smart = []
AoI_Smart = []
CPU_Smart = []
b_Smart = []
# j = param['episodes']
j = 20  # logging_timeline记录了所有EPISODE中的所有细节，所以可以放心引用
for i in range(param['num_Devices']):
    if not logging_timeline[i][j]['intervals']:
        print(i)
        continue
    KeyInterval = logging_timeline[i][j]['intervals'] + [param['nTimeUnits'] - logging_timeline[i][j]['intervals'][-1]]

    #  Reward
    KeyReward = logging_timeline[i][j]['KeyRewards']
    reward = [KeyReward[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    Reward_Smart.append(sum(reward) / param['nTimeUnits'])

    # AoI
    KeyAoI = logging_timeline[i][j]['KeyAoI']
    AoI = [KeyAoI[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    AoI_Smart.append(sum(AoI) / param['nTimeUnits'])

    # CPU
    KeyCPU = logging_timeline[i][j]['KeyCPU']
    CPU = [KeyCPU[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    CPU_Smart.append(sum(CPU) / param['nTimeUnits'])

    # b/queue
    Keyb = logging_timeline[i][j]['Keyb']
    b = [Keyb[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    b_Smart.append(sum(b) / param['nTimeUnits'])


# For UAV
UAV_Energy = [np.mean(logging_timeline[0][0]['Random_UAV_Energy']), np.mean(logging_timeline[0][0]['Force_UAV_Energy']),
          np.mean(logging_timeline[0][x]['UAV_Energy'])] # MJ, {*1000 = kJ}
UAV_R_E = [-np.mean(logging_timeline[0][0]['Random_UAV_R_E']), -np.mean(logging_timeline[0][0]['Force_UAV_R_E']),
          -np.mean(logging_timeline[0][x]['UAV_R_E'])]
UAV_Reward = [-np.mean(logging_timeline[0][0]['Random_UAV_Reward']), -np.mean(logging_timeline[0][0]['Force_UAV_Reward']),
          -np.mean(logging_timeline[0][x]['UAV_Reward'])]
UAV_AoI = [np.mean(logging_timeline[0][0]['Random_UAV_AoI']), np.mean(logging_timeline[0][0]['Force_UAV_AoI']),
          np.mean(logging_timeline[0][x]['UAV_AoI'])]
UAV_CPU = [np.mean(logging_timeline[0][0]['Random_UAV_CPU']), np.mean(logging_timeline[0][0]['Force_UAV_CPU']),
          np.mean(logging_timeline[0][x]['UAV_CPU'])]  # CYCLES
UAV_CPU_J = [CPU_J(j)*1000 for j in UAV_CPU] # mJ
UAV_b = [np.mean(logging_timeline[0][0]['Random_UAV_b']), np.mean(logging_timeline[0][0]['Force_UAV_b']),
          np.mean(logging_timeline[0][x]['UAV_b'])]

# For Devices
Device_Reward = [-mean(Reward_Random), -mean(Reward_Force), -mean(Reward_Smart)]
Device_AoI = [mean(AoI_Random), mean(AoI_Force), mean(AoI_Smart)]
Device_CPU = [mean(CPU_Random), mean(CPU_Force), mean(CPU_Smart)]
Device_b = [mean(b_Random), mean(b_Force), mean(b_Smart)]


# Start painting
fig, ax = plt.subplots()
n_groups = 3
index = np.arange(n_groups)
bar_width = 0.20
opacity = 1  #透明度，可以改成0.4
error_config = {'ecolor': '0.3'}
rects1 = plt.bar(index, UAV_Energy, bar_width,
                 alpha=opacity,
                 color='r',
                 # yerr=std_women,
                 error_kw=error_config,
                 label='Energy of System')
rects2 = plt.bar(index + bar_width, UAV_AoI, bar_width,
                 alpha=opacity,
                 color='g',
                 # yerr=std_women,
                 error_kw=error_config,
                 label='AoI of Devices')
rects3 = plt.bar(index + bar_width * 2, UAV_CPU, bar_width,
                 alpha=opacity,
                 color='y',
                 # yerr=std_women,
                 error_kw=error_config,
                 label='CPU of Devices')

# plt.xlabel('Group')
plt.ylabel('Mean Cost')
plt.title('The Comparison Between Different Methods')
plt.xticks(index + bar_width / 2, ('Random', 'Force', 'Smart'))
plt.plot(index + bar_width / 4, UAV_R_E, 'o-', label='System Cost')
plt.plot(index + bar_width / 4, UAV_Reward, '^-', label='Device Cost')
plt.legend(loc = 'upper right')
# plt.legend(loc = 'upper right',ncols=3)
axx1 = ax.twinx()
rects4 = plt.bar(index - bar_width , UAV_b, bar_width,
                 alpha=opacity,
                 color='b',
                 # yerr=std_men,
                 error_kw=error_config,
                 label='queue length of System')
plt.tight_layout()
plt.show()




d = 1