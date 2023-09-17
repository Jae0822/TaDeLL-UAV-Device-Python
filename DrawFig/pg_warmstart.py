import numpy as np
import pickle
import matplotlib.pyplot as plt
from statistics import mean

"""
基于regular与warm-start进行比较，在random. force, smart三种情况下进行了比较，
也进一步详细比较了aoi和cpu在三种方法、两种策略下的不同。
"""


# with open('fig_A19.pkl', 'rb') as f:
#     model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)



with open('./output/170923-2137-6_devices/output.pkl', 'rb') as f:
    model, nn_strategy, random_strategy, forced_strategy, param, avg, logging_timeline = pickle.load(f)


env = nn_strategy.env
env_regular = nn_strategy.env_pgrl
env_random = random_strategy.env
env_random_regular = random_strategy.env_pgrl
env_force = forced_strategy.env
env_force_regular = forced_strategy.env_pgrl

def CPU_J(cycles):
    J = pow(cycles * pow(10, 8), 3) * 4 * pow(10, -28)
    return J

def keytime2intervals(keytime):
    intervals = []
    for i in range(len(keytime) - 1):
        intervals.append(keytime[i+1] - keytime[i])
    return intervals



#  Smart
Reward_Samrt = []
Reward_Samrt_Regular = []
AoI_Smart = []
AoI_Smart_Regular = []
CPU_Smart = []
CPU_Smart_Regular = []
b_Smart = []
b_Smart_Regular = []
j = param['episodes']
# j = 20  # logging_timeline记录了所有EPISODE中的所有细节，所以可以放心引用
for i in range(param['num_Devices']):
    #  Reward
    KeyReward = logging_timeline[i][j]['KeyRewards']
    KeyReward_Regular = logging_timeline[i][j]['KeyReward_Regular']
    Reward_Samrt.append(sum(KeyReward) / param['nTimeUnits'])
    Reward_Samrt_Regular.append(sum(KeyReward_Regular) / param['nTimeUnits'])

    # AoI
    KeyAoI = logging_timeline[i][j]['KeyAoI']
    KeyAoI_Regular = logging_timeline[i][j]['KeyAoI_Regular']
    AoI_Smart.append(sum(KeyAoI) / param['nTimeUnits'])
    AoI_Smart_Regular.append(sum(KeyAoI_Regular) / param['nTimeUnits'])

    # CPU
    KeyCPU = logging_timeline[i][j]['KeyCPU']
    KeyCPU_Regular = logging_timeline[i][j]['KeyCPU_Regular']
    CPU_Smart.append(sum(KeyCPU) / param['nTimeUnits'])
    CPU_Smart_Regular.append(sum(KeyCPU_Regular) / param['nTimeUnits'])

    # b/queue
    Keyb = logging_timeline[i][j]['Keyb']
    Keyb_Regular = logging_timeline[i][j]['Keyb_Regular']
    b_Smart.append(sum(Keyb) / param['nTimeUnits'])
    b_Smart_Regular.append(sum(Keyb_Regular) / param['nTimeUnits'])


#  Random
Reward_Random = []
Reward_Random_Regular = []
AoI_Random = []
AoI_Random_Regular = []
CPU_Random = []
CPU_Random_Regular = []
b_Random = []
b_Random_Regular = []
for i in range(param['num_Devices']):

    # Reward ( = AoI + CPU)
    KeyReward = logging_timeline[i][0]['Random_KeyRewards']
    KeyReward_Regular = env_random_regular.Devices[i].KeyReward
    Reward_Random.append(sum(KeyReward) / param['nTimeUnits_random'])
    Reward_Random_Regular.append(sum(KeyReward_Regular) / param['nTimeUnits_random'])

    # AoI
    KeyAoI = logging_timeline[i][0]['Random_KeyAoI']
    KeyAoI_Regular = env_random_regular.Devices[i].KeyAoI
    AoI_Random.append(sum(KeyAoI) / param['nTimeUnits_random'])
    AoI_Random_Regular.append(sum(KeyAoI_Regular) / param['nTimeUnits_random'])

    # CPU
    KeyCPU = logging_timeline[i][0]['Random_KeyCPU']
    KeyCPU_Regular = env_random_regular.Devices[i].KeyCPU
    CPU_Random.append(sum(KeyCPU) / param['nTimeUnits_random'])
    CPU_Random_Regular.append(sum(KeyCPU_Regular) / param['nTimeUnits_random'])

    # b
    Keyb = logging_timeline[i][0]['Random_Keyb']
    Keyb_Regular = env_random_regular.Devices[i].Keyb
    b_Random.append(sum(Keyb) / param['nTimeUnits_random'])
    b_Random_Regular.append(sum(Keyb_Regular) / param['nTimeUnits_random'])




#  Force
Reward_Force = []
Reward_Force_Regular = []
AoI_Force = []
AoI_Force_Regular = []
CPU_Force = []
CPU_Force_Regular = []
b_Force = []
b_Force_Regular = []
for i in range(param['num_Devices']):

    # Reward
    KeyReward = logging_timeline[i][0]['Force_KeyRewards']
    KeyReward_Regular = env_force_regular.Devices[i].KeyReward
    Reward_Force.append(sum(KeyReward) / param['nTimeUnits_force'])
    Reward_Force_Regular.append(sum(KeyReward_Regular) / param['nTimeUnits_force'])

    # AoI
    KeyAoI = logging_timeline[i][0]['Force_KeyAoI']
    KeyAoI_Regular = env_force_regular.Devices[i].KeyAoI
    AoI_Force.append(sum(KeyAoI) / param['nTimeUnits_force'])
    AoI_Force_Regular.append(sum(KeyAoI_Regular) / param['nTimeUnits_force'])

    # CPU
    KeyCPU = logging_timeline[i][0]['Force_KeyCPU']
    KeyCPU_Regular = env_force_regular.Devices[i].KeyCPU
    CPU_Force.append(sum(KeyCPU) / param['nTimeUnits_force'])
    CPU_Force_Regular.append(sum(KeyCPU_Regular) / param['nTimeUnits_force'])

    # b
    Keyb = logging_timeline[i][0]['Force_Keyb']
    Keyb_Regular = env_force_regular.Devices[i].Keyb
    b_Force.append(sum(Keyb) / param['nTimeUnits_force'])
    b_Force_Regular.append(sum(Keyb_Regular) / param['nTimeUnits_force'])





# mean(AoI_Random) + mean(CPU_Random)
# mean(AoI_Random_Regular) + mean(CPU_Random_Regular)
# mean(AoI_Force) + mean(CPU_Force)
# mean(AoI_Force_Regular) + mean(CPU_Force_Regular)
# mean(AoI_Smart) + mean(CPU_Smart)
# mean(AoI_Smart_Regular) + mean(CPU_Smart_Regular)

fig, axs = plt.subplots(2,2, sharex=True)
fig.suptitle('The comparison between LLRL with Regular-PG, Ep:' + str(j) + ', Mean')
type = ("Random", "Force", "Smart")
x = np.arange(len(type))  # the label locations
width = 0.25  # the width of the bars


# Reward
multiplier = 0
value_means = {
    'LLRL': (mean(Reward_Random), mean(Reward_Force), mean(Reward_Samrt)),
    'PG': (mean(Reward_Random_Regular), mean(Reward_Force_Regular), mean(Reward_Samrt_Regular)),
}
for attribute, measurement in value_means.items():
    offset = width * multiplier
    rects = axs[0,0].bar(x + offset, measurement, width, label=attribute)
    # axs[0,0].bar_label(rects, padding=3)
    multiplier += 1
axs[0,0].set_ylabel('Averaged Reward')
axs[0,0].set_title('Reward of IoT Devices')
axs[0,0].set_xticks(x + width, type)
axs[0,0].legend(loc='best', ncol=2)
# axs[0,0].set_ylim(0, 2)
axs[0,0].autoscale(axis = 'y')

# AoI
multiplier = 0
value_means = {
    'LLRL': (mean(AoI_Random), mean(AoI_Force), mean(AoI_Smart)),
    'PG': (mean(AoI_Random_Regular), mean(AoI_Force_Regular), mean(AoI_Smart_Regular)),
}
for attribute, measurement in value_means.items():
    offset = width * multiplier
    rects = axs[0,1].bar(x + offset, measurement, width, label=attribute)
    # axs[0,1].bar_label(rects, padding=3)
    multiplier += 1
axs[0,1].set_ylabel('Averaged AoI')
axs[0,1].set_title('AoI of IoT Devices')
axs[0,1].set_xticks(x + width, type)
axs[0,1].legend(loc='best', ncol=2)
# axs[0,1].set_ylim(0, 100)
axs[0,1].autoscale(axis = 'y')


#  CPU
multiplier = 0
value_means = {
    'LLRL': (mean(CPU_Random), mean(CPU_Force), mean(CPU_Smart)),
    'PG': (mean(CPU_Random_Regular), mean(CPU_Force_Regular), mean(CPU_Smart_Regular)),
}
for attribute, measurement in value_means.items():
    offset = width * multiplier
    rects = axs[1,0].bar(x + offset, measurement, width, label=attribute)
    # axs[0,1].bar_label(rects, padding=3)
    multiplier += 1
axs[1,0].set_ylabel('Averaged CPU')
axs[1,0].set_title('CPU of IoT Devices')
axs[1,0].set_xticks(x + width, type)
axs[1,0].legend(loc='best', ncol=2)
# axs[1,0].set_ylim(0, 4)
axs[1,0].autoscale(axis = 'y')


#  b
multiplier = 0
value_means = {
    'LLRL': (mean(b_Random), mean(b_Force), mean(b_Smart)),
    'PG': (mean(b_Random_Regular), mean(b_Force_Regular), mean(b_Smart_Regular)),
}
for attribute, measurement in value_means.items():
    offset = width * multiplier
    rects = axs[1,1].bar(x + offset, measurement, width, label=attribute)
    # axs[0,1].bar_label(rects, padding=3)
    multiplier += 1
axs[1,1].set_ylabel('Averaged Queue')
axs[1,1].set_title('Queue of IoT Devices')
axs[1,1].set_xticks(x + width, type)
axs[1,1].legend(loc='best', ncol=2)
# axs[1,1].set_ylim(0, 100)
axs[1,1].autoscale(axis = 'y')



d = 1

# Sum 的结果


fig, axs = plt.subplots(2,2, sharex=True)
fig.suptitle('The comparison between LLRL with PG, Ep:' + str(j) + ', Sum')
type = ("Random", "Force", "Smart")
x = np.arange(len(type))  # the label locations
width = 0.25  # the width of the bars


# Reward
multiplier = 0
value_means = {
    'LLRL': (sum(Reward_Random), sum(Reward_Force), sum(Reward_Samrt)),
    'PG': (sum(Reward_Random_Regular), sum(Reward_Force_Regular), sum(Reward_Samrt_Regular)),
}
for attribute, measurement in value_means.items():
    offset = width * multiplier
    rects = axs[0,0].bar(x + offset, measurement, width, label=attribute)
    # axs[0,0].bar_label(rects, padding=3)
    multiplier += 1
axs[0,0].set_ylabel('Averaged Reward')
axs[0,0].set_title('Reward of IoT Devices')
axs[0,0].set_xticks(x + width, type)
axs[0,0].legend(loc='best', ncol=2)
# axs[0,0].set_ylim(0, 7)
axs[0,0].autoscale(axis = 'y')


# AoI
multiplier = 0
value_means = {
    'LLRL': (sum(AoI_Random), sum(AoI_Force), sum(AoI_Smart)),
    'PG': (sum(AoI_Random_Regular), sum(AoI_Force_Regular), sum(AoI_Smart_Regular)),
}
for attribute, measurement in value_means.items():
    offset = width * multiplier
    rects = axs[0,1].bar(x + offset, measurement, width, label=attribute)
    # axs[0,1].bar_label(rects, padding=3)
    multiplier += 1
axs[0,1].set_ylabel('Averaged AoI')
axs[0,1].set_title('AoI of IoT Devices')
axs[0,1].set_xticks(x + width, type)
axs[0,1].legend(loc='best', ncol=2)
# axs[0,1].set_ylim(0, 500)
axs[0,1].autoscale(axis = 'y')



#  CPU
multiplier = 0
value_means = {
    'LLRL': (sum(CPU_Random), sum(CPU_Force), sum(CPU_Smart)),
    'PG': (sum(CPU_Random_Regular), sum(CPU_Force_Regular), sum(CPU_Smart_Regular)),
}
for attribute, measurement in value_means.items():
    offset = width * multiplier
    rects = axs[1,0].bar(x + offset, measurement, width, label=attribute)
    # axs[0,1].bar_label(rects, padding=3)
    multiplier += 1
axs[1,0].set_ylabel('Averaged CPU')
axs[1,0].set_title('CPU of IoT Devices')
axs[1,0].set_xticks(x + width, type)
axs[1,0].legend(loc='best', ncol=2)
# axs[1,0].set_ylim(0, 80)
axs[1,0].autoscale(axis = 'y')


#  b
multiplier = 0
value_means = {
    'LLRL': (sum(b_Random), sum(b_Force), sum(b_Smart)),
    'PG': (sum(b_Random_Regular), sum(b_Force_Regular), sum(b_Smart_Regular)),
}
for attribute, measurement in value_means.items():
    offset = width * multiplier
    rects = axs[1,1].bar(x + offset, measurement, width, label=attribute)
    # axs[0,1].bar_label(rects, padding=3)
    multiplier += 1
axs[1,1].set_ylabel('Averaged Queue')
axs[1,1].set_title('Queue of IoT Devices')
axs[1,1].set_xticks(x + width, type)
axs[1,1].legend(loc='best', ncol=2)
# axs[1,1].set_ylim(0, 5000)
axs[1,1].autoscale(axis = 'y')

plt.show()



f = 1
