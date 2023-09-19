
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


start_ep = 400

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




# #############################################  Prepare Data  ###################################################################

UAV_AoI = [V10_UAV_AoI, V15_UAV_AoI, V20_UAV_AoI]#, V25_UAV_AoI, V30_UAV_AoI, V35_UAV_AoI_1, V40_UAV_AoI] # 因为速度快，所以访问快，所以AoI绝对值变小了
UAV_CPU = [V10_UAV_CPU, V15_UAV_CPU, V20_UAV_CPU] # V25_UAV_CPU, V30_UAV_CPU, V35_UAV_CPU_1, V40_UAV_CPU] # 因为速度快，所以访问快，所以AoI绝对值变小了
UAV_Reward = [V10_UAV_Reward, V15_UAV_Reward, V20_UAV_Reward] #, V25_UAV_Reward, V30_UAV_Reward, V35_UAV_Reward, V40_UAV_Reward] # 因为速度快，所以访问快，所以AoI绝对值变小了
# UAV_AoI_CPU = list(map(lambda x,y: x + y, UAV_AoI, UAV_CPU))
UAV_CPU_J = [pow(x * pow(10,8), 3) * 4 * pow(10, -28)  for x in UAV_CPU] # in Joule


UAV_AoI_QL = [V10_UAV_AoI_QL, V15_UAV_AoI_QL, V20_UAV_AoI_QL] #, V25_UAV_AoI_QL, V30_UAV_AoI_QL, V35_UAV_AoI_QL, V40_UAV_AoI_QL] # 因为速度快，所以访问快，所以AoI绝对值变小了
UAV_CPU_QL = [V10_UAV_CPU_QL, V15_UAV_CPU_QL, V20_UAV_CPU_QL] #, V25_UAV_CPU_QL, V30_UAV_CPU_QL, V35_UAV_CPU_QL, V40_UAV_CPU_QL] # 因为速度快，所以访问快，所以AoI绝对值变小了
UAV_Reward_QL = [V10_UAV_Reward_QL, V15_UAV_Reward_QL, V20_UAV_Reward_QL]#, V25_UAV_Reward_QL, V25_UAV_Reward_QL, V30_UAV_Reward_QL, V35_UAV_Reward_QL, V40_UAV_Reward_QL] # 因为速度快，所以访问快，所以AoI绝对值变小了
# UAV_AoI_CPU = list(map(lambda x,y: x + y, UAV_AoI, UAV_CPU))
UAV_CPU_QL_J = [pow(x * pow(10,8), 3) * 4 * pow(10, -28)  for x in UAV_CPU_QL] # in Joule


Device_AoI = [V10_Device_AoI, V15_Device_AoI, V20_Device_AoI] #, V25_Device_AoI,  V30_Device_AoI,  V35_Device_AoI,  V40_Device_AoI]
Device_CPU = [V10_Device_CPU, V15_Device_CPU, V20_Device_CPU] #, V25_Device_CPU, V30_Device_CPU, V35_Device_CPU, V40_Device_CPU]
Device_Reward = [V10_Device_Reward, V15_Device_Reward, V20_Device_Reward] #, V25_Device_Reward, V30_Device_Reward, V35_Device_Reward, V40_Device_Reward]
Device_CPU_J = [pow(x * pow(10,8), 3) * 4 * pow(10, -28)  for x in Device_CPU] # in Joule

Device_AoI_QL = [V10_Device_AoI_QL, V15_Device_AoI_QL, V20_Device_AoI_QL] #, V25_Device_AoI_QL, V30_Device_AoI_QL, V35_Device_AoI_QL, V40_Device_AoI_QL]
Device_CPU_QL = [V10_Device_CPU_QL, V15_Device_CPU_QL, V20_Device_CPU_QL] #, V25_Device_CPU_QL, V30_Device_CPU_QL, V35_Device_CPU_QL, V40_Device_CPU_QL]
Device_Reward_QL = [V10_Device_Reward_QL, V15_Device_Reward_QL, V20_Device_Reward_QL] #, V25_Device_Reward_QL, V30_Device_Reward_QL, V35_Device_Reward_QL, V40_Device_Reward_QL]
Device_CPU_QL_J = [pow(x * pow(10,8), 3) * 4 * pow(10, -28)  for x in Device_CPU_QL] # in Joule


# #############################################  画图  ###################################################################

end_V = 25

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

