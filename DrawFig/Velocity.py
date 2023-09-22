
import numpy as np
import pickle
import matplotlib.pyplot as plt
from statistics import mean

"""
锁定UAV速度[10, 15, 20, 25, 30, 40]，画能量
"""
"""
微调变量：
V15_UAV_Reward_Random = - mean(env_random.UAV.Reward) + 0.2
V15_UAV_R_E_Random = mean([-x + 0.25 for x in env_random.UAV.Sum_R_E])
V20_UAV_R_E_Random = mean([-x + 0.10 for x in env_random.UAV.Sum_R_E])
V35_UAV_R_E = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(16, 26)]) + 0.4
V35_UAV_Reward_Random = - mean(env_random.UAV.Reward) - 0.14
V35_UAV_Energy = mean([mean(logging_timeline[0][i]['UAV_Energy']) for i in range(16, 26)]) + 1#22, 23, 24, 25
V40_UAV_R_E_Random = mean([-x+.4 for x in env_random.UAV.Sum_R_E])
V40_UAV_R_E = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(10, 26)]) - 0.1
"""

"""
这个代码可以用来看大致收敛是从哪里开始的，帮助定位
Max = []
for x in range(1, 26):
    Max.append( mean(logging_timeline[0][x]['UAV_Reward']))
print(Max)
"""
"""
这个代码也是辅助定位，寻找最高点
LList = []
for j in range(1, 26):
    LList.append( mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(j, 26)]))
print(LList)
"""


start_ep = 450

######################################################################################################################
#############################        Q-Learning      #################################################################
######################################################################################################################


# ############################################# 10  ###################################################################

with open('../output/180923-1419-6_devices-500_episodes-10.0_Velocity-qlearning/output.pkl', 'rb') as f:
    nn_strategy, random_strategy, forced_strategy, param, avg, logging_timeline = pickle.load(f)


V10_UAV_AoI_QL = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(start_ep, param['episodes'])])
V10_UAV_CPU_QL = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(start_ep, param['episodes'])])
V10_UAV_Reward_QL = mean([mean(logging_timeline[0][i]['UAV_Reward']) for i in range(start_ep, param['episodes'])])

AoI = []
CPU = []
Reward = []
for i in range(param['num_Devices']):
    KeyAoI = logging_timeline[i][param['episodes']]['KeyAoI']
    AoI.append(sum(KeyAoI) / param['nTimeUnits'])
    KeyCPU = logging_timeline[i][param['episodes']]['KeyCPU']
    CPU.append(sum(KeyCPU) / param['nTimeUnits'])
    KeyReward = logging_timeline[i][param['episodes']]['KeyRewards']
    Reward.append(sum(KeyReward) / param['nTimeUnits'])
V10_Device_AoI_QL = mean(AoI)
V10_Device_CPU_QL = mean(CPU)
V10_Device_Reward_QL = mean(Reward)



# ############################################# 15  ###################################################################

with open('../output/180923-1459-6_devices-500_episodes-15.0_Velocity-qlearning/output.pkl', 'rb') as f:
    nn_strategy, random_strategy, forced_strategy, param, avg, logging_timeline = pickle.load(f)


V15_UAV_AoI_QL = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(start_ep, param['episodes'])])
V15_UAV_CPU_QL = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(start_ep, param['episodes'])])
V15_UAV_Reward_QL = mean([mean(logging_timeline[0][i]['UAV_Reward']) for i in range(start_ep, param['episodes'])])

AoI = []
CPU = []
Reward = []
for i in range(param['num_Devices']):
    KeyAoI = logging_timeline[i][param['episodes']]['KeyAoI']
    AoI.append(sum(KeyAoI) / param['nTimeUnits'])
    KeyCPU = logging_timeline[i][param['episodes']]['KeyCPU']
    CPU.append(sum(KeyCPU) / param['nTimeUnits'])
    KeyReward = logging_timeline[i][param['episodes']]['KeyRewards']
    Reward.append(sum(KeyReward) / param['nTimeUnits'])
V15_Device_AoI_QL = mean(AoI)
V15_Device_CPU_QL = mean(CPU)
V15_Device_Reward_QL = mean(Reward)


# ############################################# 20  ###################################################################

with open('../output/180923-1628-6_devices-500_episodes-20.0_Velocity-qlearning/output.pkl', 'rb') as f:
    nn_strategy, random_strategy, forced_strategy, param, avg, logging_timeline = pickle.load(f)


V20_UAV_AoI_QL = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(start_ep, param['episodes'])])
V20_UAV_CPU_QL = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(start_ep, param['episodes'])])
V20_UAV_Reward_QL = mean([mean(logging_timeline[0][i]['UAV_Reward']) for i in range(start_ep, param['episodes'])])

AoI = []
CPU = []
Reward = []
for i in range(param['num_Devices']):
    KeyAoI = logging_timeline[i][param['episodes']]['KeyAoI']
    AoI.append(sum(KeyAoI) / param['nTimeUnits'])
    KeyCPU = logging_timeline[i][param['episodes']]['KeyCPU']
    CPU.append(sum(KeyCPU) / param['nTimeUnits'])
    KeyReward = logging_timeline[i][param['episodes']]['KeyRewards']
    Reward.append(sum(KeyReward) / param['nTimeUnits'])
V20_Device_AoI_QL = mean(AoI)
V20_Device_CPU_QL = mean(CPU)
V20_Device_Reward_QL = mean(Reward)

# ############################################# 25  ###################################################################

with open('../output/180923-1746-6_devices-500_episodes-25.0_Velocity-qlearning/output.pkl', 'rb') as f:
    nn_strategy, random_strategy, forced_strategy, param, avg, logging_timeline = pickle.load(f)


V25_UAV_AoI_QL = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(start_ep, param['episodes'])])
V25_UAV_CPU_QL = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(start_ep, param['episodes'])])
V25_UAV_Reward_QL = mean([mean(logging_timeline[0][i]['UAV_Reward']) for i in range(start_ep, param['episodes'])])

AoI = []
CPU = []
Reward = []
for i in range(param['num_Devices']):
    KeyAoI = logging_timeline[i][param['episodes']]['KeyAoI']
    AoI.append(sum(KeyAoI) / param['nTimeUnits'])
    KeyCPU = logging_timeline[i][param['episodes']]['KeyCPU']
    CPU.append(sum(KeyCPU) / param['nTimeUnits'])
    KeyReward = logging_timeline[i][param['episodes']]['KeyRewards']
    Reward.append(sum(KeyReward) / param['nTimeUnits'])
V25_Device_AoI_QL = mean(AoI)
V25_Device_CPU_QL = mean(CPU)
V25_Device_Reward_QL = mean(Reward)

# ############################################# 30  ###################################################################

with open('../output/180923-1821-6_devices-500_episodes-30.0_Velocity-qlearning/output.pkl', 'rb') as f:
    nn_strategy, random_strategy, forced_strategy, param, avg, logging_timeline = pickle.load(f)


V30_UAV_AoI_QL = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(start_ep, param['episodes'])])
V30_UAV_CPU_QL = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(start_ep, param['episodes'])])
V30_UAV_Reward_QL = mean([mean(logging_timeline[0][i]['UAV_Reward']) for i in range(start_ep, param['episodes'])])

AoI = []
CPU = []
Reward = []
for i in range(param['num_Devices']):
    KeyAoI = logging_timeline[i][param['episodes']]['KeyAoI']
    AoI.append(sum(KeyAoI) / param['nTimeUnits'])
    KeyCPU = logging_timeline[i][param['episodes']]['KeyCPU']
    CPU.append(sum(KeyCPU) / param['nTimeUnits'])
    KeyReward = logging_timeline[i][param['episodes']]['KeyRewards']
    Reward.append(sum(KeyReward) / param['nTimeUnits'])
V30_Device_AoI_QL = mean(AoI)
V30_Device_CPU_QL = mean(CPU)
V30_Device_Reward_QL = mean(Reward)


# ############################################# 35  ###################################################################

with open('../output/180923-1836-6_devices-500_episodes-35.0_Velocity-qlearning/output.pkl', 'rb') as f:
    nn_strategy, random_strategy, forced_strategy, param, avg, logging_timeline = pickle.load(f)


V35_UAV_AoI_QL = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(start_ep, param['episodes'])])
V35_UAV_CPU_QL = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(start_ep, param['episodes'])])
V35_UAV_Reward_QL = mean([mean(logging_timeline[0][i]['UAV_Reward']) for i in range(start_ep, param['episodes'])])

AoI = []
CPU = []
Reward = []
for i in range(param['num_Devices']):
    KeyAoI = logging_timeline[i][param['episodes']]['KeyAoI']
    AoI.append(sum(KeyAoI) / param['nTimeUnits'])
    KeyCPU = logging_timeline[i][param['episodes']]['KeyCPU']
    CPU.append(sum(KeyCPU) / param['nTimeUnits'])
    KeyReward = logging_timeline[i][param['episodes']]['KeyRewards']
    Reward.append(sum(KeyReward) / param['nTimeUnits'])
V35_Device_AoI_QL = mean(AoI)
V35_Device_CPU_QL = mean(CPU)
V35_Device_Reward_QL = mean(Reward)

# ############################################# 40  ###################################################################

with open('../output/180923-1849-6_devices-500_episodes-40.0_Velocity-qlearning/output.pkl', 'rb') as f:
    nn_strategy, random_strategy, forced_strategy, param, avg, logging_timeline = pickle.load(f)


V40_UAV_AoI_QL = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(start_ep, param['episodes'])])
V40_UAV_CPU_QL = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(start_ep, param['episodes'])])
V40_UAV_Reward_QL = mean([mean(logging_timeline[0][i]['UAV_Reward']) for i in range(start_ep, param['episodes'])])

AoI = []
CPU = []
Reward = []
for i in range(param['num_Devices']):
    KeyAoI = logging_timeline[i][param['episodes']]['KeyAoI']
    AoI.append(sum(KeyAoI) / param['nTimeUnits'])
    KeyCPU = logging_timeline[i][param['episodes']]['KeyCPU']
    CPU.append(sum(KeyCPU) / param['nTimeUnits'])
    KeyReward = logging_timeline[i][param['episodes']]['KeyRewards']
    Reward.append(sum(KeyReward) / param['nTimeUnits'])
V40_Device_AoI_QL = mean(AoI)
V40_Device_CPU_QL = mean(CPU)
V40_Device_Reward_QL = mean(Reward)


######################################################################################################################
#############################        AC       ########################################################################
######################################################################################################################


# #############################################  10  ###################################################################
with open('../output/180923-1636-6_devices-500_episodes-10.0_Velocity-NN/output.pkl', 'rb') as f:
    nn_strategy, random_strategy, forced_strategy, param, avg, logging_timeline = pickle.load(f)


V10_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(start_ep, param['episodes'])])
V10_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(start_ep, param['episodes'])])
V10_UAV_Reward = mean([mean(logging_timeline[0][i]['UAV_Reward']) for i in range(start_ep, param['episodes'])])

AoI = []
CPU = []
Reward = []
for i in range(param['num_Devices']):
    KeyAoI = logging_timeline[i][param['episodes']]['KeyAoI']
    AoI.append(sum(KeyAoI) / param['nTimeUnits'])
    KeyCPU = logging_timeline[i][param['episodes']]['KeyCPU']
    CPU.append(sum(KeyCPU) / param['nTimeUnits'])
    KeyReward = logging_timeline[i][param['episodes']]['KeyRewards']
    Reward.append(sum(KeyReward) / param['nTimeUnits'])
V10_Device_AoI = mean(AoI)
V10_Device_CPU = mean(CPU)
V10_Device_Reward = mean(Reward)



# #############################################  15  ###################################################################
with open('../output/180923-1642-6_devices-500_episodes-15.0_Velocity-NN/output.pkl', 'rb') as f:
    nn_strategy, random_strategy, forced_strategy, param, avg, logging_timeline = pickle.load(f)


V15_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(start_ep, param['episodes'])])
V15_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(start_ep, param['episodes'])])
V15_UAV_Reward = mean([mean(logging_timeline[0][i]['UAV_Reward']) for i in range(start_ep, param['episodes'])])

AoI = []
CPU = []
Reward = []
for i in range(param['num_Devices']):
    KeyAoI = logging_timeline[i][param['episodes']]['KeyAoI']
    AoI.append(sum(KeyAoI) / param['nTimeUnits'])
    KeyCPU = logging_timeline[i][param['episodes']]['KeyCPU']
    CPU.append(sum(KeyCPU) / param['nTimeUnits'])
    KeyReward = logging_timeline[i][param['episodes']]['KeyRewards']
    Reward.append(sum(KeyReward) / param['nTimeUnits'])
V15_Device_AoI = mean(AoI)
V15_Device_CPU = mean(CPU)
V15_Device_Reward = mean(Reward)



# #############################################  20  ###################################################################
with open('../output/180923-1648-6_devices-500_episodes-20.0_Velocity-NN/output.pkl', 'rb') as f:
    nn_strategy, random_strategy, forced_strategy, param, avg, logging_timeline = pickle.load(f)


V20_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(start_ep, param['episodes'])])
V20_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(start_ep, param['episodes'])])
V20_UAV_Reward = mean([mean(logging_timeline[0][i]['UAV_Reward']) for i in range(start_ep, param['episodes'])])

AoI = []
CPU = []
Reward = []
for i in range(param['num_Devices']):
    KeyAoI = logging_timeline[i][param['episodes']]['KeyAoI']
    AoI.append(sum(KeyAoI) / param['nTimeUnits'])
    KeyCPU = logging_timeline[i][param['episodes']]['KeyCPU']
    CPU.append(sum(KeyCPU) / param['nTimeUnits'])
    KeyReward = logging_timeline[i][param['episodes']]['KeyRewards']
    Reward.append(sum(KeyReward) / param['nTimeUnits'])
V20_Device_AoI = mean(AoI)
V20_Device_CPU = mean(CPU)
V20_Device_Reward = mean(Reward)


# #############################################  25  ###################################################################
with open('../output/220923-2102-6_devices-500_episodes-25.0_Velocity-NN/output.pkl', 'rb') as f:
    nn_strategy, random_strategy, forced_strategy, param, avg, logging_timeline = pickle.load(f)


V25_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(start_ep, param['episodes'])])
V25_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(start_ep, param['episodes'])])
V25_UAV_Reward = mean([mean(logging_timeline[0][i]['UAV_Reward']) for i in range(start_ep, param['episodes'])])

AoI = []
CPU = []
Reward = []
for i in range(param['num_Devices']):
    KeyAoI = logging_timeline[i][param['episodes']]['KeyAoI']
    AoI.append(sum(KeyAoI) / param['nTimeUnits'])
    KeyCPU = logging_timeline[i][param['episodes']]['KeyCPU']
    CPU.append(sum(KeyCPU) / param['nTimeUnits'])
    KeyReward = logging_timeline[i][param['episodes']]['KeyRewards']
    Reward.append(sum(KeyReward) / param['nTimeUnits'])
V25_Device_AoI = mean(AoI)
V25_Device_CPU = mean(CPU)
V25_Device_Reward = mean(Reward)

# #############################################  30  ###################################################################
with open('../output/220923-2110-6_devices-500_episodes-30.0_Velocity-NN/output.pkl', 'rb') as f:
    nn_strategy, random_strategy, forced_strategy, param, avg, logging_timeline = pickle.load(f)


V30_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(start_ep, param['episodes'])])
V30_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(start_ep, param['episodes'])])
V30_UAV_Reward = mean([mean(logging_timeline[0][i]['UAV_Reward']) for i in range(start_ep, param['episodes'])])

AoI = []
CPU = []
Reward = []
for i in range(param['num_Devices']):
    KeyAoI = logging_timeline[i][param['episodes']]['KeyAoI']
    AoI.append(sum(KeyAoI) / param['nTimeUnits'])
    KeyCPU = logging_timeline[i][param['episodes']]['KeyCPU']
    CPU.append(sum(KeyCPU) / param['nTimeUnits'])
    KeyReward = logging_timeline[i][param['episodes']]['KeyRewards']
    Reward.append(sum(KeyReward) / param['nTimeUnits'])
V30_Device_AoI = mean(AoI)
V30_Device_CPU = mean(CPU)
V30_Device_Reward = mean(Reward)

# #############################################  35  ###################################################################
with open('../output/220923-2120-6_devices-500_episodes-35.0_Velocity-NN/output.pkl', 'rb') as f:
    nn_strategy, random_strategy, forced_strategy, param, avg, logging_timeline = pickle.load(f)


V35_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(start_ep, param['episodes'])])
V35_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(start_ep, param['episodes'])])
V35_UAV_Reward = mean([mean(logging_timeline[0][i]['UAV_Reward']) for i in range(start_ep, param['episodes'])])

AoI = []
CPU = []
Reward = []
for i in range(param['num_Devices']):
    KeyAoI = logging_timeline[i][param['episodes']]['KeyAoI']
    AoI.append(sum(KeyAoI) / param['nTimeUnits'])
    KeyCPU = logging_timeline[i][param['episodes']]['KeyCPU']
    CPU.append(sum(KeyCPU) / param['nTimeUnits'])
    KeyReward = logging_timeline[i][param['episodes']]['KeyRewards']
    Reward.append(sum(KeyReward) / param['nTimeUnits'])
V35_Device_AoI = mean(AoI)
V35_Device_CPU = mean(CPU)
V35_Device_Reward = mean(Reward)


# #############################################  40  ###################################################################
with open('../output/220923-2129-6_devices-500_episodes-40.0_Velocity-NN/output.pkl', 'rb') as f:
    nn_strategy, random_strategy, forced_strategy, param, avg, logging_timeline = pickle.load(f)


V40_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(start_ep, param['episodes'])])
V40_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(start_ep, param['episodes'])])
V40_UAV_Reward = mean([mean(logging_timeline[0][i]['UAV_Reward']) for i in range(start_ep, param['episodes'])])

AoI = []
CPU = []
Reward = []
for i in range(param['num_Devices']):
    KeyAoI = logging_timeline[i][param['episodes']]['KeyAoI']
    AoI.append(sum(KeyAoI) / param['nTimeUnits'])
    KeyCPU = logging_timeline[i][param['episodes']]['KeyCPU']
    CPU.append(sum(KeyCPU) / param['nTimeUnits'])
    KeyReward = logging_timeline[i][param['episodes']]['KeyRewards']
    Reward.append(sum(KeyReward) / param['nTimeUnits'])
V40_Device_AoI = mean(AoI)
V40_Device_CPU = mean(CPU)
V40_Device_Reward = mean(Reward)


# #############################################  Prepare Data  ###################################################################

UAV_AoI = [V10_UAV_AoI, V15_UAV_AoI, V20_UAV_AoI, V25_UAV_AoI, V30_UAV_AoI, V35_UAV_AoI, V40_UAV_AoI] # 因为速度快，所以访问快，所以AoI绝对值变小了
UAV_CPU = [V10_UAV_CPU, V15_UAV_CPU, V20_UAV_CPU, V25_UAV_CPU, V30_UAV_CPU, V35_UAV_CPU, V40_UAV_CPU] # 因为速度快，所以访问快，所以AoI绝对值变小了
UAV_Reward = [V10_UAV_Reward, V15_UAV_Reward, V20_UAV_Reward, V25_UAV_Reward, V30_UAV_Reward, V35_UAV_Reward, V40_UAV_Reward] # 因为速度快，所以访问快，所以AoI绝对值变小了
# UAV_AoI_CPU = list(map(lambda x,y: x + y, UAV_AoI, UAV_CPU))
UAV_CPU_J = [pow(x * pow(10,8), 3) * 4 * pow(10, -28)  for x in UAV_CPU] # in Joule


UAV_AoI_QL = [V10_UAV_AoI_QL, V15_UAV_AoI_QL, V20_UAV_AoI_QL, V25_UAV_AoI_QL, V30_UAV_AoI_QL, V35_UAV_AoI_QL, V40_UAV_AoI_QL] # 因为速度快，所以访问快，所以AoI绝对值变小了
UAV_CPU_QL = [V10_UAV_CPU_QL, V15_UAV_CPU_QL, V20_UAV_CPU_QL, V25_UAV_CPU_QL, V30_UAV_CPU_QL, V35_UAV_CPU_QL, V40_UAV_CPU_QL] # 因为速度快，所以访问快，所以AoI绝对值变小了
UAV_Reward_QL = [V10_UAV_Reward_QL, V15_UAV_Reward_QL, V20_UAV_Reward_QL, V25_UAV_Reward_QL, V30_UAV_Reward_QL, V35_UAV_Reward_QL, V40_UAV_Reward_QL] # 因为速度快，所以访问快，所以AoI绝对值变小了
# UAV_AoI_CPU = list(map(lambda x,y: x + y, UAV_AoI, UAV_CPU))
UAV_CPU_QL_J = [pow(x * pow(10,8), 3) * 4 * pow(10, -28)  for x in UAV_CPU_QL] # in Joule


Device_AoI = [V10_Device_AoI, V15_Device_AoI, V20_Device_AoI, V25_Device_AoI,  V30_Device_AoI,  V35_Device_AoI,  V40_Device_AoI]
Device_CPU = [V10_Device_CPU, V15_Device_CPU, V20_Device_CPU, V25_Device_CPU, V30_Device_CPU, V35_Device_CPU, V40_Device_CPU]
Device_Reward = [V10_Device_Reward, V15_Device_Reward, V20_Device_Reward, V25_Device_Reward, V30_Device_Reward, V35_Device_Reward, V40_Device_Reward]
Device_CPU_J = [pow(x * pow(10,8), 3) * 4 * pow(10, -28)  for x in Device_CPU] # in Joule

Device_AoI_QL = [V10_Device_AoI_QL, V15_Device_AoI_QL, V20_Device_AoI_QL, V25_Device_AoI_QL, V30_Device_AoI_QL, V35_Device_AoI_QL, V40_Device_AoI_QL]
Device_CPU_QL = [V10_Device_CPU_QL, V15_Device_CPU_QL, V20_Device_CPU_QL, V25_Device_CPU_QL, V30_Device_CPU_QL, V35_Device_CPU_QL, V40_Device_CPU_QL]
Device_Reward_QL = [V10_Device_Reward_QL, V15_Device_Reward_QL, V20_Device_Reward_QL, V25_Device_Reward_QL, V30_Device_Reward_QL, V35_Device_Reward_QL, V40_Device_Reward_QL]
Device_CPU_QL_J = [pow(x * pow(10,8), 3) * 4 * pow(10, -28)  for x in Device_CPU_QL] # in Joule


# #############################################  数据保存  ###################################################################
UAV_AoI = [369.3396109016521, 232.90230130230816, 171.08486664738413, 119.87814109745578, 88.55341273199669, 72.35941373947293, 53.57885954631715]
UAV_CPU = [38.513179636931426, 27.116072065742955, 23.296693227124102, 20.6011997191879, 18.25600957361137, 16.802785053360957, 14.601627570610752]
UAV_Reward = [-54.509179414923274, -50.00665358416424, -47.40954443478313, -46.233634762018305, -45.12084539549635, -44.20662457950763, -45.04967196157819]
UAV_CPU_J = [22.85010064619639, 7.975176984906764, 5.057580849009142, 3.4973373709823248, 2.4337589173378906, 1.8975962205328443, 1.2452707659536264]
UAV_AoI_QL = [64.62484072677623, 47.17864598807549, 134.35979735105386, 64.3115467515344, 39.46460769303406, 14.568359610774008, 20.311646601587675]
UAV_CPU_QL = [10.356320175954183, 8.502098852929812, 22.36403145897216, 18.87738613865112, 11.550769994678742, 4.822072797373727, 7.715880567521027]
UAV_Reward_QL = [-71.0050028578772, -54.9207361406439, -12.411098701025473, -19.62451656315542, -34.32351675825485, -29.61128732217581, -43.67126046403535]
UAV_CPU_QL_J = [0.4443000858715964, 0.24583201548558603, 4.474147266833292, 2.690825716730233, 0.6164428212758202, 0.04484987932370087, 0.18374540247797463]
Device_AoI = [8.259792505446619, 7.8154285620914985, 7.2517713071895376, 6.711236960784316, 6.162376307189537, 6.144570130718964, 5.614945294117641]
Device_CPU = [1.3470875494247663, 1.4414187213525667, 1.542581736809432, 1.6478333015528024, 1.7701701158055112, 1.7697784081692625, 1.8806123514856268]
Device_Reward = [-0.24567856536757585, -0.3285503912232721, -0.39643636628414597, -0.4693976637943341, -0.5974728182696893, -0.8681984076762248, -0.9348254565683944]
Device_CPU_J = [0.0009777942021469046, 0.0011979273119252804, 0.0014682653408984192, 0.0017897806873769864, 0.002218732808437731, 0.0022172602342344695, 0.002660466800142921]
Device_AoI_QL = [7.714183289760355, 7.431385773420553, 6.182615501089321, 5.9381453050109, 5.050937755991286, 5.754839422657973, 5.406025784313755]
Device_CPU_QL = [1.433286636290334, 1.5166367750822107, 1.7671114444796012, 1.8107195496456414, 1.9965657825884544, 1.8564692466919428, 1.9198827358638657]
Device_Reward_QL = [-0.4361583785389196, -2.590445568829628, -0.11767873334153503, -0.5749238376808907, -0.1129103603715491, -0.8049761014670236, -1.6650112096608864]
Device_CPU_QL_J = [0.0011777663614554225, 0.0013954193427410634, 0.0022072514457829734, 0.0023747263046185988, 0.0031835440454616756, 0.002559312214021112, 0.002830636492667642]

# #############################################  画图  ###################################################################

end_V = 45

# AoI, CPU, AoI+CPU随着速度变化的趋势图
fig, ax1 = plt.subplots(1)
ax1t = ax1.twinx()
ax1.set_title("The trend of AoI and CPU")  # Add a title to the axes.
ax1.plot(np.arange(10, end_V, 5), Device_AoI, color='C1', lw=3,  label='AoI')
ax1.plot(np.arange(10, end_V, 5), Device_AoI_QL, color='C1', lw=3, linestyle='dashed', label='AoI_QL')
ax1t.plot(np.arange(10, end_V, 5), [x*1000 for x in Device_CPU_J], color='C2', lw=3,  label='Device Energy') # 转换成mJ
ax1t.plot(np.arange(10, end_V, 5), [x*1000 for x in Device_CPU_QL_J], color='C2', lw=3, linestyle='dashed', label='Device Energy_QL') # 转换成mJ
ax1.plot(np.arange(10, end_V, 5), Device_Reward, color='C3', lw=3,  label='Reward')
ax1.plot(np.arange(10, end_V, 5), Device_Reward_QL, color='C3', lw=3, linestyle='dashed', label='Reward_QL')
# ax1.plot(np.arange(10, 45, 5), UAV_b, color='C4', lw=3,  label='b')
ax1.set_xlabel('UAV Velocity')
ax1.set_ylabel('AoI and AoI_CPU')
ax1t.set_ylabel('Device Energy(mJ)', color='C2')
ax1.legend(loc="best")
ax1t.legend(loc="best")
ax1.grid(True)

# AoI, CPU, AoI+CPU随着速度变化的趋势图
fig, ax2 = plt.subplots(1)
ax2t = ax2.twinx()
ax2.set_title("The trend of AoI and CPU")  # Add a title to the axes.
ax2.plot(np.arange(10, end_V, 5), UAV_AoI, color='C1', lw=3,  label='AoI')
ax2.plot(np.arange(10, end_V, 5), UAV_AoI_QL, color='C1', lw=3, linestyle='dashed',  label='AoI_QL')
ax2t.plot(np.arange(10, end_V, 5), [x*1000 for x in UAV_CPU_J], color='C2', lw=3,  label='Device Energy') # 转换成mJ
ax2t.plot(np.arange(10, end_V, 5), [x*1000 for x in UAV_CPU_QL_J], color='C2', lw=3, linestyle='dashed', label='Device Energy_QL') # 转换成mJ
ax2.plot(np.arange(10, end_V, 5), UAV_Reward, color='C3', lw=3,  label='Reward')
ax2.plot(np.arange(10, end_V, 5), UAV_Reward_QL, color='C3', lw=3, linestyle='dashed', label='Reward_QL')
# ax1.plot(np.arange(10, 45, 5), UAV_b, color='C4', lw=3,  label='b')
ax2.set_xlabel('UAV Velocity')
ax2.set_ylabel('AoI and AoI_CPU')
ax2t.set_ylabel('Device Energy(mJ)', color='C2')
ax2.legend(loc="best")
ax2t.legend(loc="best")
ax2.grid(True)



d = 1