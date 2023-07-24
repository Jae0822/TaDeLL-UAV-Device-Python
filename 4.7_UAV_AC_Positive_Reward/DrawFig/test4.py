import numpy as np
import pickle
import matplotlib.pyplot as plt
from statistics import mean

"""
基于regular与warm-start进行比较，在random. force, smart三种情况下进行了比较，
也进一步详细比较了aoi和cpu在三种方法、两种策略下的不同。
"""


with open('fig_A19.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)





#  Random
Reward_Random = []
Reward_Random_Regular = []
AoI_Random = []
AoI_Random_Regular = []
CPU_Random = []
CPU_Random_Regular = []
for i in range(param['num_Devices']):
    KeyInterval = env_random.Devices[i].intervals + [param['nTimeUnits_random'] - env_random.Devices[i].KeyTime[-1]]

    # Reward ( = AoI + CPU)
    KeyReward = logging_timeline[i][0]['Random_KeyRewards']
    KeyReward_Regular = env_random.Devices[i].KeyReward_Regular
    reward = [KeyReward[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    Reward_Random.append(sum(reward) / param['nTimeUnits_random'])
    reward_Regular = [KeyReward_Regular[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    Reward_Random_Regular.append(sum(reward_Regular) / param['nTimeUnits_random'])

    # AoI
    KeyAoI = logging_timeline[i][0]['Random_KeyAoI']
    KeyAoI_Regular = env_random.Devices[i].KeyAoI_Regular
    AoI = [KeyAoI[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    AoI_Random.append(sum(AoI) / param['nTimeUnits_random'])
    AoI_Regular = [KeyAoI_Regular[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    AoI_Random_Regular.append(sum(AoI_Regular) / param['nTimeUnits_random'])

    # CPU
    KeyCPU = logging_timeline[i][0]['Random_KeyCPU']
    KeyCPU_Regular = env_random.Devices[i].KeyCPU_Regular
    CPU = [KeyCPU[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    CPU_Random.append(sum(CPU) / param['nTimeUnits_random'])
    CPU_Regular = [KeyCPU_Regular[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    CPU_Random_Regular.append(sum(CPU_Regular) / param['nTimeUnits_random'])






#  Force
Reward_Force = []
Reward_Force_Regular = []
AoI_Force = []
AoI_Force_Regular = []
CPU_Force = []
CPU_Force_Regular = []
for i in range(param['num_Devices']):
    KeyInterval = env_force.Devices[i].intervals + [param['nTimeUnits_force'] - env_force.Devices[i].KeyTime[-1]]

    # Reward
    KeyReward = logging_timeline[i][0]['Force_KeyRewards']
    KeyReward_Regular = env_force.Devices[i].KeyReward_Regular
    reward = [KeyReward[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    Reward_Force.append(sum(reward) / param['nTimeUnits_force'])
    reward_Regular = [KeyReward_Regular[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    Reward_Force_Regular.append(sum(reward_Regular) / param['nTimeUnits_force'])

    # AoI
    KeyAoI = logging_timeline[i][0]['Force_KeyAoI']
    KeyAoI_Regular = env_force.Devices[i].KeyAoI_Regular
    AoI = [KeyAoI[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    AoI_Force.append(sum(AoI) / param['nTimeUnits_force'])
    AoI_Regular = [KeyAoI_Regular[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    AoI_Force_Regular.append(sum(AoI_Regular) / param['nTimeUnits_force'])

    # CPU
    KeyCPU = logging_timeline[i][0]['Force_KeyCPU']
    KeyCPU_Regular = env_force.Devices[i].KeyCPU_Regular
    CPU = [KeyCPU[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    CPU_Force.append(sum(CPU) / param['nTimeUnits_force'])
    CPU_Regular = [KeyCPU_Regular[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    CPU_Force_Regular.append(sum(CPU_Regular) / param['nTimeUnits_force'])






#  Smart
Reward_Samrt = []
Reward_Samrt_Regular = []
AoI_Smart = []
AoI_Smart_Regular = []
CPU_Smart = []
CPU_Smart_Regular = []
j = param['episodes']  # 因为logging_timeline中没有记录每一个DEVICE的每一个EPISODE的REGULAR数据
# j = 12
for i in range(param['num_Devices']):
    KeyInterval = env.Devices[i].intervals + [param['nTimeUnits'] - env.Devices[i].KeyTime[-1]]

    #  Reward
    KeyReward = logging_timeline[i][j]['KeyRewards']
    KeyReward_Regular = env.Devices[i].KeyReward_Regular
    reward = [KeyReward[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    Reward_Samrt.append(sum(reward) / param['nTimeUnits'])
    reward_Regular = [KeyReward_Regular[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    Reward_Samrt_Regular.append(sum(reward_Regular) / param['nTimeUnits'])

    # AoI
    KeyAoI = logging_timeline[i][j]['KeyAoI']
    KeyAoI_Regular = env.Devices[i].KeyAoI_Regular
    AoI = [KeyAoI[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    AoI_Smart.append(sum(AoI) / param['nTimeUnits'])
    AoI_Regular = [KeyAoI_Regular[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    AoI_Smart_Regular.append(sum(AoI_Regular) / param['nTimeUnits'])

    # CPU
    KeyCPU = logging_timeline[i][j]['KeyCPU']
    KeyCPU_Regular = env.Devices[i].KeyCPU_Regular
    CPU = [KeyCPU[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    CPU_Smart.append(sum(CPU) / param['nTimeUnits'])
    CPU_Regular = [KeyCPU_Regular[x] * KeyInterval[x] for x in range(len(KeyInterval))]
    CPU_Smart_Regular.append(sum(CPU_Regular) / param['nTimeUnits'])


mean(AoI_Random) + mean(CPU_Random)
mean(AoI_Random_Regular) + mean(CPU_Random_Regular)
mean(AoI_Force) + mean(CPU_Force)
mean(AoI_Force_Regular) + mean(CPU_Force_Regular)
mean(AoI_Smart) + mean(CPU_Smart)
mean(AoI_Smart_Regular) + mean(CPU_Smart_Regular)

fig, axs = plt.subplots(2,2, sharex=True)
fig.suptitle('The comparison between LLRL with Regular-PG')
type = ("Random", "Force", "Smart")
x = np.arange(len(type))  # the label locations
width = 0.25  # the width of the bars


# Reward
multiplier = 0
value_means = {
    'LLRL': (-mean(Reward_Random), -mean(Reward_Force), -mean(Reward_Samrt)),
    'PG': (-mean(Reward_Random_Regular), -mean(Reward_Force_Regular), -mean(Reward_Samrt_Regular)),
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
axs[0,0].set_ylim(0, 7)

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
axs[0,1].set_ylim(0, 12)

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
axs[1,0].set_ylim(0, 2)

#  AoI + CPU
multiplier = 0
value_means = {
    'LLRL': (mean(CPU_Random), mean(CPU_Force), mean(CPU_Smart)),
    'PG': (mean(CPU_Random_Regular), mean(CPU_Force_Regular), mean(CPU_Smart_Regular)),
}
for attribute, measurement in value_means.items():
    offset = width * multiplier
    rects = axs[1,1].bar(x + offset, measurement, width, label=attribute)
    # axs[0,1].bar_label(rects, padding=3)
    multiplier += 1
axs[1,1].set_ylabel('Averaged CPU')
axs[1,1].set_title('CPU of IoT Devices')
axs[1,1].set_xticks(x + width, type)
axs[1,1].legend(loc='best', ncol=2)
axs[1,1].set_ylim(0, 5)



d = 1

#
# # 横向三幅图
# fig, axs = plt.subplots(1,3, sharex=True)
# fig.suptitle('The comparison between LLRL with Regular-PG')
# type = ("Random", "Force", "Smart")
# x = np.arange(len(type))  # the label locations
# width = 0.25  # the width of the bars
#
#
# # Reward
# multiplier = 0
# value_means = {
#     'LLRL': (-mean(Reward_Random), -mean(Reward_Force), -mean(Reward_Samrt)),
#     'PG': (-mean(Reward_Random_Regular), -mean(Reward_Force_Regular), -mean(Reward_Samrt_Regular)),
# }
# for attribute, measurement in value_means.items():
#     offset = width * multiplier
#     rects = axs[0].bar(x + offset, measurement, width, label=attribute)
#     # axs[0,0].bar_label(rects, padding=3)
#     multiplier += 1
# axs[0].set_ylabel('Averaged Reward')
# axs[0].set_title('Reward of IoT Devices')
# axs[0].set_xticks(x + width, type)
# axs[0].legend(loc='best', ncol=2)
# axs[0].set_ylim(0, 7)
#
# # AoI
# multiplier = 0
# value_means = {
#     'LLRL': (mean(AoI_Random), mean(AoI_Force), mean(AoI_Smart)),
#     'PG': (mean(AoI_Random_Regular), mean(AoI_Force_Regular), mean(AoI_Smart_Regular)),
# }
# for attribute, measurement in value_means.items():
#     offset = width * multiplier
#     rects = axs[1].bar(x + offset, measurement, width, label=attribute)
#     # axs[0,1].bar_label(rects, padding=3)
#     multiplier += 1
# axs[1].set_ylabel('Averaged AoI')
# axs[1].set_title('AoI of IoT Devices')
# axs[1].set_xticks(x + width, type)
# axs[1].legend(loc='best', ncol=2)
# axs[1].set_ylim(0, 12)
#
# #  CPU
# multiplier = 0
# value_means = {
#     'LLRL': (mean(CPU_Random), mean(CPU_Force), mean(CPU_Smart)),
#     'PG': (mean(CPU_Random_Regular), mean(CPU_Force_Regular), mean(CPU_Smart_Regular)),
# }
# for attribute, measurement in value_means.items():
#     offset = width * multiplier
#     rects = axs[2].bar(x + offset, measurement, width, label=attribute)
#     # axs[0,1].bar_label(rects, padding=3)
#     multiplier += 1
# axs[2].set_ylabel('Averaged CPU')
# axs[2].set_title('CPU of IoT Devices')
# axs[2].set_xticks(x + width, type)
# axs[2].legend(loc='best', ncol=2)
# axs[2].set_ylim(0, 2)