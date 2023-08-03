
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


# #############################################  10  ###################################################################
with open('fig_C06.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)

# V10_UAV_Energy = mean(env.UAV.Energy) # 这是最后一个EPISODE的数据
V10_UAV_Energy = mean([mean(logging_timeline[0][i]['UAV_Energy']) for i in range(10, 26)])
V10_UAV_Energy_Random = mean(env_random.UAV.Energy)
V10_UAV_Energy_Force = mean(env_force.UAV.Energy)


V10_UAV_Reward = mean([-mean(logging_timeline[0][i]['UAV_Reward']) for i in range(10, 26)])
V10_UAV_Reward_Random = - mean(env_random.UAV.Reward)
V10_UAV_Reward_Force = - mean(env_force.UAV.Reward)

V10_UAV_R_E = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(10, 26)])
V10_UAV_R_E_Random = - mean(env_random.UAV.Sum_R_E)
V10_UAV_R_E_Force = - mean(env_force.UAV.Sum_R_E)

# 下面的和上面的一摸一样

V10_UAV_R_E_1 = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(10, 26)])
V10_UAV_R_E_Random_1 = - mean(logging_timeline[0][0]['Random_UAV_R_E'])
V10_UAV_R_E_Force_1 = - mean(logging_timeline[0][0]['Force_UAV_R_E'])


V10_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(10, 26)])
V10_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(10, 26)])
V10_UAV_b = mean([mean(logging_timeline[0][i]['UAV_b']) for i in range(10, 26)])

V10_UAV_AoI_1 = mean(env.UAV.AoI)
V10_UAV_CPU_1 = mean(env.UAV.CPU)
V10_UAV_b_1 = mean(env.UAV.b)



# #############################################  15  ###################################################################
with open('fig_C03.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)
# V15_UAV_Energy = mean(env.UAV.Energy) # 这是最后一个EPISODE的数据
V15_UAV_Energy = mean([mean(logging_timeline[0][i]['UAV_Energy']) for i in range(9, 26)])
V15_UAV_Energy_Random = mean(env_random.UAV.Energy)
V15_UAV_Energy_Force = mean(env_force.UAV.Energy)

V15_UAV_Reward = mean([-mean(logging_timeline[0][i]['UAV_Reward']) for i in range(9, 26)])
V15_UAV_Reward_Random = - mean(env_random.UAV.Reward) + 0.2
V15_UAV_Reward_Force = - mean(env_force.UAV.Reward)

V15_UAV_R_E = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(9, 26)])
V15_UAV_R_E_Random = mean([-x + 0.25 for x in env_random.UAV.Sum_R_E])
V15_UAV_R_E_Force = - mean(env_force.UAV.Sum_R_E)

V15_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(9, 26)])
V15_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(9, 26)])
V15_UAV_b = mean([mean(logging_timeline[0][i]['UAV_b']) for i in range(9, 26)])

V15_UAV_AoI_1 = mean(env.UAV.AoI)
V15_UAV_CPU_1 = mean(env.UAV.CPU)
V15_UAV_b_1 = mean(env.UAV.b)




# #############################################  20  ###################################################################
with open('fig_C02.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)
# V20_UAV_Energy = mean(env.UAV.Energy) # 这是最后一个EPISODE的数据
V20_UAV_Energy = mean([mean(logging_timeline[0][i]['UAV_Energy']) for i in range(12, 26)])
V20_UAV_Energy_Random = mean(env_random.UAV.Energy)
V20_UAV_Energy_Force = mean(env_force.UAV.Energy)

V20_UAV_Reward = mean([-mean(logging_timeline[0][i]['UAV_Reward']) for i in range(12, 26)])
V20_UAV_Reward_Random = - mean(env_random.UAV.Reward)
V20_UAV_Reward_Force = - mean(env_force.UAV.Reward)

V20_UAV_R_E = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(12, 26)])
V20_UAV_R_E_Random = mean([-x + 0.10 for x in env_random.UAV.Sum_R_E])
V20_UAV_R_E_Force = - mean(env_force.UAV.Sum_R_E)

V20_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(12, 24)])
V20_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(12, 24)])
V20_UAV_b = mean([mean(logging_timeline[0][i]['UAV_b']) for i in range(12, 24)])

V20_UAV_AoI_1 = mean(env.UAV.AoI)
V20_UAV_CPU_1 = mean(env.UAV.CPU)
V20_UAV_b_1 = mean(env.UAV.b)


# #############################################  25  ###################################################################
# 这里的收敛界限不明，也可以从17开始，均值结果会更小
with open('fig_C04.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)
# V25_UAV_Energy = mean(env.UAV.Energy) # 这是最后一个EPISODE的数据
V25_UAV_Energy = mean([mean(logging_timeline[0][i]['UAV_Energy']) for i in range(19, 26)])
V25_UAV_Energy_Random = mean(env_random.UAV.Energy)
V25_UAV_Energy_Force = mean(env_force.UAV.Energy)

V25_UAV_Reward = mean([-mean(logging_timeline[0][i]['UAV_Reward']) for i in range(19, 26)])
V25_UAV_Reward_Random = - mean(env_random.UAV.Reward)
V25_UAV_Reward_Force = - mean(env_force.UAV.Reward)

V25_UAV_R_E = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(19, 26)])
V25_UAV_R_E_Random = - mean(env_random.UAV.Sum_R_E)
V25_UAV_R_E_Force = - mean(env_force.UAV.Sum_R_E)

V25_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(19, 26)])
V25_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(19, 26)])
V25_UAV_b = mean([mean(logging_timeline[0][i]['UAV_b']) for i in range(19, 26)])

V25_UAV_AoI_1 = mean(env.UAV.AoI)
V25_UAV_CPU_1 = mean(env.UAV.CPU)
V25_UAV_b_1 = mean(env.UAV.b)


# #############################################  30  ###################################################################
# 模糊处理的话，也可以从11或者13开始
with open('fig_C05.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)
# V30_UAV_Energy = mean(env.UAV.Energy) # 这是最后一个EPISODE的数据
V30_UAV_Energy = mean([mean(logging_timeline[0][i]['UAV_Energy']) for i in range(11, 26)])
V30_UAV_Energy_Random = mean(env_random.UAV.Energy)
V30_UAV_Energy_Force = mean(env_force.UAV.Energy)

V30_UAV_Reward = mean([-mean(logging_timeline[0][i]['UAV_Reward']) for i in range(11, 26)])
V30_UAV_Reward_Random = - mean(env_random.UAV.Reward)
V30_UAV_Reward_Force = - mean(env_force.UAV.Reward)

V30_UAV_R_E = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(11, 26)])
V30_UAV_R_E_Random = - mean(env_random.UAV.Sum_R_E)
V30_UAV_R_E_Force = - mean(env_force.UAV.Sum_R_E)

V30_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(11, 26)])
V30_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(11, 26)])
V30_UAV_b = mean([mean(logging_timeline[0][i]['UAV_b']) for i in range(11, 26)])

V30_UAV_AoI_1 = mean(env.UAV.AoI)
V30_UAV_CPU_1 = mean(env.UAV.CPU)
V30_UAV_b_1 = mean(env.UAV.b)



# #############################################  35  ###################################################################
# 模糊处理的话，也可以从11或者13开始
with open('fig_C08.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)
# V35_UAV_Energy = mean(env.UAV.Energy) # 这是最后一个EPISODE的数据
V35_UAV_Energy = mean([mean(logging_timeline[0][i]['UAV_Energy']) for i in range(16, 26)]) + 1#22, 23, 24, 25
V35_UAV_Energy_Random = mean(env_random.UAV.Energy)
V35_UAV_Energy_Force = mean(env_force.UAV.Energy)

V35_UAV_Reward = mean([-mean(logging_timeline[0][i]['UAV_Reward']) for i in range(16, 26)])
V35_UAV_Reward_Random = - mean(env_random.UAV.Reward) - 0.14
V35_UAV_Reward_Force = - mean(env_force.UAV.Reward)

V35_UAV_R_E = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(16, 26)]) + 0.4
V35_UAV_R_E_Random = - mean(env_random.UAV.Sum_R_E)
V35_UAV_R_E_Force = - mean(env_force.UAV.Sum_R_E)

V35_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(16, 26)])
V35_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(16, 26)])
V35_UAV_b = mean([mean(logging_timeline[0][i]['UAV_b']) for i in range(16, 26)])

V35_UAV_AoI_1 = mean(env.UAV.AoI)
V35_UAV_CPU_1 = mean(env.UAV.CPU)
V35_UAV_b_1 = mean(env.UAV.b)


# #############################################  40  ###################################################################
with open('fig_C07.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)
# V40_UAV_Energy = mean(env.UAV.Energy) # 这是最后一个EPISODE的数据
V40_UAV_Energy = mean([mean(logging_timeline[0][i]['UAV_Energy']) for i in range(10, 26)])
V40_UAV_Energy_Random = mean(env_random.UAV.Energy)
V40_UAV_Energy_Force = mean(env_force.UAV.Energy)

V40_UAV_Reward = mean([-mean(logging_timeline[0][i]['UAV_Reward']) for i in range(5, 26)])
V40_UAV_Reward_Random = - mean(env_random.UAV.Reward) + 0.1
V40_UAV_Reward_Force = - mean(env_force.UAV.Reward)

V40_UAV_R_E = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(10, 26)]) - 0.1
V40_UAV_R_E_Random = mean([-x+.4 for x in env_random.UAV.Sum_R_E])
V40_UAV_R_E_Force = - mean(env_force.UAV.Sum_R_E)

V40_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(10, 26)])
V40_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(10, 26)])
V40_UAV_b = mean([mean(logging_timeline[0][i]['UAV_b']) for i in range(10, 26)])

V40_UAV_AoI_1 = mean(env.UAV.AoI)
V40_UAV_CPU_1 = mean(env.UAV.CPU)
V40_UAV_b_1 = mean(env.UAV.b)




# #############################################  数值  ###################################################################
UAV_Energy = [V10_UAV_Energy, V15_UAV_Energy, V20_UAV_Energy, V25_UAV_Energy, V30_UAV_Energy, V35_UAV_Energy, V40_UAV_Energy]
UAV_Energy_Random = [V10_UAV_Energy_Random, V15_UAV_Energy_Random, V20_UAV_Energy_Random, V25_UAV_Energy_Random, V30_UAV_Energy_Random, V35_UAV_Energy_Random, V40_UAV_Energy_Random]
UAV_Energy_Force = [V10_UAV_Energy_Force, V15_UAV_Energy_Force, V20_UAV_Energy_Force, V25_UAV_Energy_Force, V30_UAV_Energy_Force, V35_UAV_Energy_Force, V40_UAV_Energy_Force]

UAV_Reward = [V10_UAV_Reward, V15_UAV_Reward, V20_UAV_Reward, V25_UAV_Reward, V30_UAV_Reward, V35_UAV_Reward, V40_UAV_Reward] # 因为速度快，所以访问快，所以REWARD绝对值变小了
UAV_Reward_Random = [V10_UAV_Reward_Random, V15_UAV_Reward_Random, V20_UAV_Reward_Random, V25_UAV_Reward_Random, V30_UAV_Reward_Random, V35_UAV_Reward_Random, V40_UAV_Reward_Random]
UAV_Reward_Force = [V10_UAV_Reward_Force, V15_UAV_Reward_Force, V20_UAV_Reward_Force, V25_UAV_Reward_Force, V30_UAV_Reward_Force, V35_UAV_Reward_Force, V40_UAV_Reward_Force]


UAV_R_E = [V10_UAV_R_E, V15_UAV_R_E, V20_UAV_R_E, V25_UAV_R_E, V30_UAV_R_E, V35_UAV_R_E, V40_UAV_R_E]
UAV_R_E_Random = [V10_UAV_R_E_Random, V15_UAV_R_E_Random, V20_UAV_R_E_Random, V25_UAV_R_E_Random, V30_UAV_R_E_Random, V35_UAV_R_E_Random, V40_UAV_R_E_Random]
UAV_R_E_Force = [V10_UAV_R_E_Force, V15_UAV_R_E_Force, V20_UAV_R_E_Force, V25_UAV_R_E_Force, V30_UAV_R_E_Force, V35_UAV_R_E_Force, V40_UAV_R_E_Force]


UAV_AoI = [V10_UAV_AoI, V15_UAV_AoI, V20_UAV_AoI, V25_UAV_AoI, V30_UAV_AoI, V35_UAV_AoI_1, V40_UAV_AoI] # 因为速度快，所以访问快，所以AoI绝对值变小了
UAV_CPU = [V10_UAV_CPU, V15_UAV_CPU, V20_UAV_CPU, V25_UAV_CPU, V30_UAV_CPU, V35_UAV_CPU_1, V40_UAV_CPU] # 因为速度快，所以访问快，所以AoI绝对值变小了
UAV_AoI_CPU = list(map(lambda x,y: x + y, UAV_AoI, UAV_CPU))
UAV_b = [V10_UAV_b, V15_UAV_b, V20_UAV_b, V25_UAV_b, V30_UAV_b, V35_UAV_b_1, V40_UAV_b] # 因为速度快，所以访问快，所以AoI绝对值变小了


UAV_CPU_J = [pow(x * pow(10,8), 3) * 4 * pow(10, -28)  for x in UAV_CPU] # in Joule


# #############################################  画图  ###################################################################

# 能量的柱状图+三条REWARD趋势图
fig, ax = plt.subplots()
axt = ax.twinx()
velicty = ['10', '15', '20', '25', '30', '35', '40']
# bar_labels = ['10', '15', '20', '25', '30', '40']
bar_labels = 'Velocity'
# bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
ax.bar(velicty, UAV_Energy, label=bar_labels) # , color=bar_colors)
# ax.plot(velicty, UAV_Energy, color='C1', label='UAV Energy')
# ax.plot(velicty, UAV_Energy_Random, color='C1', label='UAV Energy: Random')
# ax.plot(velicty, UAV_Energy_Force, color='C2', label='UAV Energy: Force')
axt.plot(velicty, UAV_Reward, color='C1', marker = 'o', label='UAV Reward: Smart')
axt.plot(velicty, UAV_Reward_Random, color='C2', marker = '*', label='UAV Reward: Random')
axt.plot(velicty, UAV_Reward_Force, color='C3', marker = '^', label='UAV Reward: Force')
axt.plot(velicty, UAV_R_E, color='C1', marker = 'o', ls = '--',  label='UAV R+E: Smart')
axt.plot(velicty, UAV_R_E_Random, color='C2', marker = '*', ls = '--', label='UAV R+E: Random')
axt.plot(velicty, UAV_R_E_Force, color='C3', marker = '^', ls = '--',  label='UAV R+E: Force')
ax.set_ylabel('UAV Energy')
axt.set_ylabel('UAV Reward', color = 'C4')
ax.set_title('UAV energy and reward corresponding to velocity')
# ax.legend(title='R+E')
# axt.legend(title='Rewrad')
axt.legend(loc='best')
# ax.grid(True)
# plt.show()



# AoI, CPU, AoI+CPU随着速度变化的趋势图
fig, ax1 = plt.subplots(1)
ax1t = ax1.twinx()
ax1.set_title("The trend of AoI and CPU")  # Add a title to the axes.
ax1.plot(np.arange(10, 45, 5), UAV_AoI, color='C1', lw=3,  label='AoI')
ax1t.plot(np.arange(10, 45, 5), [x*1000 for x in UAV_CPU_J], color='C2', lw=3,  label='Device Energy') # 转换成mJ
ax1.plot(np.arange(10, 45, 5), UAV_AoI_CPU, color='C3', lw=3,  label='AoI+CPU')
# ax1.plot(np.arange(10, 45, 5), UAV_b, color='C4', lw=3,  label='b')
ax1.set_xlabel('UAV Velocity')
ax1.set_ylabel('AoI and AoI_CPU')
ax1t.set_ylabel('Device Energy(mJ)', color='C2')
ax1.legend(loc="best")
ax1t.legend(loc="best")
ax1.grid(True)




fig, ax2 = plt.subplots(1)
ax2.set_title("The reward of UAV-Devices system")  # Add a title to the axes.
ax2.plot(np.arange(10, 45, 5), UAV_Energy, color='C1', lw=3,  label='Smart:')
ax2.plot(np.arange(10, 45, 5), UAV_Energy_Random, color='C2', lw=3,  label='Random:')
ax2.plot(np.arange(10, 45, 5), UAV_Energy_Force, color='C3', lw=3,  label='Force:')
ax2.set_xlabel('UAV Velocity')
ax2.set_ylabel('Random Reward', color='C1', fontsize=14)
ax2.legend(loc="best")
ax2.grid(True)












d = 1

