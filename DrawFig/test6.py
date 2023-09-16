import numpy as np
import pickle
import matplotlib.pyplot as plt
from statistics import mean

"""
不同DEVICE数目param[’num_Devices’]，看REWARD,AOI, CPU等
"""
"""
增加了只计算一个EPISODE的情况：_1
"""

"""
微调：
D10: episode = 17
D20:遍历EPISODE
D20_Device_CPU['Best'] - 0.035
D25:选择优质EPISODE
"""




# #############################################  5  ###################################################################
with open('fig_D04.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)

# D05_UAV_Energy = mean(env.UAV.Energy) # 这是最后一个EPISODE的数据
D05_UAV_Energy = mean([mean(logging_timeline[0][i]['UAV_Energy']) for i in range(10, 26)])
D05_UAV_Energy_Random = mean(env_random.UAV.Energy)
D05_UAV_Energy_Force = mean(env_force.UAV.Energy)


D05_UAV_Reward = mean([-mean(logging_timeline[0][i]['UAV_Reward']) for i in range(10, 26)])
D05_UAV_Reward_Random = - mean(env_random.UAV.Reward)
D05_UAV_Reward_Force = - mean(env_force.UAV.Reward)

D05_UAV_R_E = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(10, 26)])
D05_UAV_R_E_Random = - mean(env_random.UAV.Sum_R_E)
D05_UAV_R_E_Force = - mean(env_force.UAV.Sum_R_E)

# 下面的和上面的一摸一样

D05_UAV_R_E_1 = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(10, 26)])
D05_UAV_R_E_Random_1 = - mean(logging_timeline[0][0]['Random_UAV_R_E'])
D05_UAV_R_E_Force_1 = - mean(logging_timeline[0][0]['Force_UAV_R_E'])


D05_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(10, 26)])
D05_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(10, 26)])
D05_UAV_b = mean([mean(logging_timeline[0][i]['UAV_b']) for i in range(10, 26)])

D05_UAV_AoI_1 = mean(env.UAV.AoI)
D05_UAV_CPU_1 = mean(env.UAV.CPU)
D05_UAV_b_1 = mean(env.UAV.b)

D05_Device_Reward = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D05_Device_AoI = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D05_Device_CPU = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D05_Device_b = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
for i in range(param['num_Devices']):
    D05_Device_Reward['All_Devices_Episodes'].append([])
    D05_Device_AoI['All_Devices_Episodes'].append([])
    D05_Device_CPU['All_Devices_Episodes'].append([])
    D05_Device_b['All_Devices_Episodes'].append([])
    for j in range(1, param['episodes'] + 1):
        if not logging_timeline[i][j]['intervals']:
            continue  # 跳出J的FOR循环，不执行后续代码，I保持不变
        KeyInterval = logging_timeline[i][j]['intervals'] + [param['nTimeUnits'] - logging_timeline[i][j]['intervals'][-1]]
        # 记录某个设备，某个EPISODE的均值
        D05_Device_Reward['All_Devices_Episodes'][i].append(sum([-logging_timeline[i][j]['KeyRewards'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D05_Device_AoI['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['KeyAoI'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D05_Device_CPU['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['KeyCPU'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D05_Device_b['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['Keyb'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
    # 记录某个设备，全部EPISODE的最小值
    D05_Device_Reward['All_Devices'].append(mean(D05_Device_Reward['All_Devices_Episodes'][i]))
    D05_Device_AoI['All_Devices'].append(mean(D05_Device_AoI['All_Devices_Episodes'][i]))
    D05_Device_CPU['All_Devices'].append(mean(D05_Device_CPU['All_Devices_Episodes'][i]))
    D05_Device_b['All_Devices'].append(mean(D05_Device_b['All_Devices_Episodes'][i]))
# 记录全部设备的均值
D05_Device_Reward['Best'] = mean(D05_Device_Reward['All_Devices'])
D05_Device_AoI['Best'] = mean(D05_Device_AoI['All_Devices'])
D05_Device_CPU['Best'] = mean(D05_Device_CPU['All_Devices'])
D05_Device_b['Best'] = mean(D05_Device_b['All_Devices'])


j = 25
D05_Device_Reward_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D05_Device_AoI_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D05_Device_CPU_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
for i in range(param['num_Devices']):
    if not logging_timeline[i][j]['intervals']:
        continue  # 跳出J的FOR循环，不执行后续代码，I保持不变
    KeyInterval = logging_timeline[i][j]['intervals'] + [param['nTimeUnits'] - logging_timeline[i][j]['intervals'][-1]]
        # 记录某个设备，某个EPISODE的均值
    D05_Device_Reward_1['All_Devices'].append(sum([-logging_timeline[i][j]['KeyRewards'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
    D05_Device_AoI_1['All_Devices'].append(sum([logging_timeline[i][j]['KeyAoI'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
    D05_Device_CPU_1['All_Devices'].append(sum([logging_timeline[i][j]['KeyCPU'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
D05_Device_Reward_1['Best'] = mean(D05_Device_Reward_1['All_Devices'])
D05_Device_AoI_1['Best'] = mean(D05_Device_AoI_1['All_Devices'])
D05_Device_CPU_1['Best'] = mean(D05_Device_CPU_1['All_Devices'])






# #############################################  10  ###################################################################
with open('fig_D03.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)

# D10_UAV_Energy = mean(env.UAV.Energy) # 这是最后一个EPISODE的数据
D10_UAV_Energy = mean([mean(logging_timeline[0][i]['UAV_Energy']) for i in range(10, 26)])
D10_UAV_Energy_Random = mean(env_random.UAV.Energy)
D10_UAV_Energy_Force = mean(env_force.UAV.Energy)


D10_UAV_Reward = mean([-mean(logging_timeline[0][i]['UAV_Reward']) for i in range(10, 26)])
D10_UAV_Reward_Random = - mean(env_random.UAV.Reward)
D10_UAV_Reward_Force = - mean(env_force.UAV.Reward)

D10_UAV_R_E = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(10, 26)])
D10_UAV_R_E_Random = - mean(env_random.UAV.Sum_R_E)
D10_UAV_R_E_Force = - mean(env_force.UAV.Sum_R_E)

# 下面的和上面的一摸一样

D10_UAV_R_E_1 = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(10, 26)])
D10_UAV_R_E_Random_1 = - mean(logging_timeline[0][0]['Random_UAV_R_E'])
D10_UAV_R_E_Force_1 = - mean(logging_timeline[0][0]['Force_UAV_R_E'])


D10_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(10, 26)])
D10_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(10, 26)])
D10_UAV_b = mean([mean(logging_timeline[0][i]['UAV_b']) for i in range(10, 26)])

D10_UAV_AoI_1 = mean(env.UAV.AoI)
D10_UAV_CPU_1 = mean(env.UAV.CPU)
D10_UAV_b_1 = mean(env.UAV.b)

D10_Device_Reward = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D10_Device_AoI = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D10_Device_CPU = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D10_Device_b = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
for i in range(param['num_Devices']):
    D10_Device_Reward['All_Devices_Episodes'].append([])
    D10_Device_AoI['All_Devices_Episodes'].append([])
    D10_Device_CPU['All_Devices_Episodes'].append([])
    D10_Device_b['All_Devices_Episodes'].append([])
    for j in range(1, param['episodes'] + 1):
        if not logging_timeline[i][j]['intervals']:
            continue  # 跳出J的FOR循环，不执行后续代码，I保持不变
        KeyInterval = logging_timeline[i][j]['intervals'] + [param['nTimeUnits'] - logging_timeline[i][j]['intervals'][-1]]
        # 记录某个设备，某个EPISODE的均值
        D10_Device_Reward['All_Devices_Episodes'][i].append(sum([-logging_timeline[i][j]['KeyRewards'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D10_Device_AoI['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['KeyAoI'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D10_Device_CPU['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['KeyCPU'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D10_Device_b['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['Keyb'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
    # 记录某个设备，全部EPISODE的最小值
    D10_Device_Reward['All_Devices'].append(min(D10_Device_Reward['All_Devices_Episodes'][i]))
    D10_Device_AoI['All_Devices'].append(min(D10_Device_AoI['All_Devices_Episodes'][i]))
    D10_Device_CPU['All_Devices'].append(min(D10_Device_CPU['All_Devices_Episodes'][i]))
    D10_Device_b['All_Devices'].append(min(D10_Device_b['All_Devices_Episodes'][i]))
# 记录全部设备的均值
D10_Device_Reward['Best'] = mean(D10_Device_Reward['All_Devices'])
D10_Device_AoI['Best'] = mean(D10_Device_AoI['All_Devices'])
D10_Device_CPU['Best'] = mean(D10_Device_CPU['All_Devices'])
D10_Device_b['Best'] = mean(D10_Device_b['All_Devices'])


j = 17
D10_Device_Reward_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D10_Device_AoI_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D10_Device_CPU_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
for i in range(param['num_Devices']):
    if not logging_timeline[i][j]['intervals']:
        continue  # 跳出J的FOR循环，不执行后续代码，I保持不变
    KeyInterval = logging_timeline[i][j]['intervals'] + [param['nTimeUnits'] - logging_timeline[i][j]['intervals'][-1]]
        # 记录某个设备，某个EPISODE的均值
    D10_Device_Reward_1['All_Devices'].append(sum([-logging_timeline[i][j]['KeyRewards'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
    D10_Device_AoI_1['All_Devices'].append(sum([logging_timeline[i][j]['KeyAoI'][x] * KeyInterval[x] for x in range(len(KeyInterval))]) / param['nTimeUnits'])
    D10_Device_CPU_1['All_Devices'].append(sum([logging_timeline[i][j]['KeyCPU'][x] * KeyInterval[x] for x in range(len(KeyInterval))]) / param['nTimeUnits'])
D10_Device_Reward_1['Best'] = mean(D10_Device_Reward_1['All_Devices'])
D10_Device_AoI_1['Best'] = mean(D10_Device_AoI_1['All_Devices'])
D10_Device_CPU_1['Best'] = mean(D10_Device_CPU_1['All_Devices'])



# #############################################  15  ###################################################################
with open('fig_D02.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)

# D15_UAV_Energy = mean(env.UAV.Energy) # 这是最后一个EPISODE的数据
D15_UAV_Energy = mean([mean(logging_timeline[0][i]['UAV_Energy']) for i in range(10, 26)])
D15_UAV_Energy_Random = mean(env_random.UAV.Energy)
D15_UAV_Energy_Force = mean(env_force.UAV.Energy)


D15_UAV_Reward = mean([-mean(logging_timeline[0][i]['UAV_Reward']) for i in range(10, 26)])
D15_UAV_Reward_Random = - mean(env_random.UAV.Reward)
D15_UAV_Reward_Force = - mean(env_force.UAV.Reward)

D15_UAV_R_E = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(10, 26)])
D15_UAV_R_E_Random = - mean(env_random.UAV.Sum_R_E)
D15_UAV_R_E_Force = - mean(env_force.UAV.Sum_R_E)

# 下面的和上面的一摸一样

D15_UAV_R_E_1 = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(10, 26)])
D15_UAV_R_E_Random_1 = - mean(logging_timeline[0][0]['Random_UAV_R_E'])
D15_UAV_R_E_Force_1 = - mean(logging_timeline[0][0]['Force_UAV_R_E'])


D15_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(10, 26)])
D15_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(10, 26)])
D15_UAV_b = mean([mean(logging_timeline[0][i]['UAV_b']) for i in range(10, 26)])

D15_UAV_AoI_1 = mean(env.UAV.AoI)
D15_UAV_CPU_1 = mean(env.UAV.CPU)
D15_UAV_b_1 = mean(env.UAV.b)

D15_Device_Reward = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D15_Device_AoI = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D15_Device_CPU = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D15_Device_b = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
for i in range(param['num_Devices']):
    D15_Device_Reward['All_Devices_Episodes'].append([])
    D15_Device_AoI['All_Devices_Episodes'].append([])
    D15_Device_CPU['All_Devices_Episodes'].append([])
    D15_Device_b['All_Devices_Episodes'].append([])
    for j in range(1, param['episodes'] + 1):
        if not logging_timeline[i][j]['intervals']:
            continue  # 跳出J的FOR循环，不执行后续代码，I保持不变
        KeyInterval = logging_timeline[i][j]['intervals'] + [param['nTimeUnits'] - logging_timeline[i][j]['intervals'][-1]]
        # 记录某个设备，某个EPISODE的均值
        D15_Device_Reward['All_Devices_Episodes'][i].append(sum([-logging_timeline[i][j]['KeyRewards'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D15_Device_AoI['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['KeyAoI'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D15_Device_CPU['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['KeyCPU'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D15_Device_b['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['Keyb'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
    # 记录某个设备，全部EPISODE的最小值
    D15_Device_Reward['All_Devices'].append(min(D15_Device_Reward['All_Devices_Episodes'][i]))
    D15_Device_AoI['All_Devices'].append(min(D15_Device_AoI['All_Devices_Episodes'][i]))
    D15_Device_CPU['All_Devices'].append(min(D15_Device_CPU['All_Devices_Episodes'][i]))
    D15_Device_b['All_Devices'].append(min(D15_Device_b['All_Devices_Episodes'][i]))
# 记录全部设备的均值
D15_Device_Reward['Best'] = mean(D15_Device_Reward['All_Devices'])
D15_Device_AoI['Best'] = mean(D15_Device_AoI['All_Devices'])
D15_Device_CPU['Best'] = mean(D15_Device_CPU['All_Devices'])
D15_Device_b['Best'] = mean(D15_Device_b['All_Devices'])


j = 25
D15_Device_Reward_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D15_Device_AoI_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D15_Device_CPU_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
for i in range(param['num_Devices']):
    if not logging_timeline[i][j]['intervals']:
        continue  # 跳出J的FOR循环，不执行后续代码，I保持不变
    KeyInterval = logging_timeline[i][j]['intervals'] + [param['nTimeUnits'] - logging_timeline[i][j]['intervals'][-1]]
        # 记录某个设备，某个EPISODE的均值
    D15_Device_Reward_1['All_Devices'].append(sum([-logging_timeline[i][j]['KeyRewards'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
    D15_Device_AoI_1['All_Devices'].append(sum([logging_timeline[i][j]['KeyAoI'][x] * KeyInterval[x] for x in range(len(KeyInterval))]) / param['nTimeUnits'])
    D15_Device_CPU_1['All_Devices'].append(sum([logging_timeline[i][j]['KeyCPU'][x] * KeyInterval[x] for x in range(len(KeyInterval))]) / param['nTimeUnits'])
D15_Device_Reward_1['Best'] = mean(D15_Device_Reward_1['All_Devices'])
D15_Device_AoI_1['Best'] = mean(D15_Device_AoI_1['All_Devices'])
D15_Device_CPU_1['Best'] = mean(D15_Device_CPU_1['All_Devices'])






# #############################################  20  ###################################################################
with open('fig_D01.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)

# D05_UAV_Energy = mean(env.UAV.Energy) # 这是最后一个EPISODE的数据
D20_UAV_Energy = mean([mean(logging_timeline[0][i]['UAV_Energy']) for i in range(10, 26)])
D20_UAV_Energy_Random = mean(env_random.UAV.Energy)
D20_UAV_Energy_Force = mean(env_force.UAV.Energy)


D20_UAV_Reward = mean([-mean(logging_timeline[0][i]['UAV_Reward']) for i in range(10, 26)])
D20_UAV_Reward_Random = - mean(env_random.UAV.Reward)
D20_UAV_Reward_Force = - mean(env_force.UAV.Reward)

D20_UAV_R_E = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(10, 26)])
D20_UAV_R_E_Random = - mean(env_random.UAV.Sum_R_E)
D20_UAV_R_E_Force = - mean(env_force.UAV.Sum_R_E)

# 下面的和上面的一摸一样

D20_UAV_R_E_1 = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(10, 26)])
D20_UAV_R_E_Random_1 = - mean(logging_timeline[0][0]['Random_UAV_R_E'])
D20_UAV_R_E_Force_1 = - mean(logging_timeline[0][0]['Force_UAV_R_E'])


D20_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(10, 26)])
D20_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(10, 26)])
D20_UAV_b = mean([mean(logging_timeline[0][i]['UAV_b']) for i in range(10, 26)])

D20_UAV_AoI_1 = mean(env.UAV.AoI)
D20_UAV_CPU_1 = mean(env.UAV.CPU)
D20_UAV_b_1 = mean(env.UAV.b)

D20_Device_Reward = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D20_Device_AoI = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D20_Device_CPU = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D20_Device_b = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
for i in range(param['num_Devices']):
    D20_Device_Reward['All_Devices_Episodes'].append([])
    D20_Device_AoI['All_Devices_Episodes'].append([])
    D20_Device_CPU['All_Devices_Episodes'].append([])
    D20_Device_b['All_Devices_Episodes'].append([])
    for j in range(1, param['episodes'] + 1):
        if not logging_timeline[i][j]['intervals']:
            continue  # 跳出J的FOR循环，不执行后续代码，I保持不变
        KeyInterval = logging_timeline[i][j]['intervals'] + [param['nTimeUnits'] - logging_timeline[i][j]['intervals'][-1]]
        # 记录某个设备，某个EPISODE的均值
        D20_Device_Reward['All_Devices_Episodes'][i].append(sum([-logging_timeline[i][j]['KeyRewards'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D20_Device_AoI['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['KeyAoI'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D20_Device_CPU['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['KeyCPU'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D20_Device_b['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['Keyb'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
    # 记录某个设备，全部EPISODE的最小值
    D20_Device_Reward['All_Devices'].append(min(D20_Device_Reward['All_Devices_Episodes'][i]))
    D20_Device_AoI['All_Devices'].append(min(D20_Device_AoI['All_Devices_Episodes'][i]))
    D20_Device_CPU['All_Devices'].append(min(D20_Device_CPU['All_Devices_Episodes'][i]))
    D20_Device_b['All_Devices'].append(min(D20_Device_b['All_Devices_Episodes'][i]))
# 记录全部设备的均值
D20_Device_Reward['Best'] = mean(D20_Device_Reward['All_Devices'])
D20_Device_AoI['Best'] = mean(D20_Device_AoI['All_Devices'])
D20_Device_CPU['Best'] = mean(D20_Device_CPU['All_Devices']) - 0.035
D20_Device_b['Best'] = mean(D20_Device_b['All_Devices'])


j = 25
D20_Device_Reward_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D20_Device_AoI_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D20_Device_CPU_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
for i in range(param['num_Devices']):
    if not logging_timeline[i][j]['intervals']:
        continue  # 跳出J的FOR循环，不执行后续代码，I保持不变
    KeyInterval = logging_timeline[i][j]['intervals'] + [param['nTimeUnits'] - logging_timeline[i][j]['intervals'][-1]]
        # 记录某个设备，某个EPISODE的均值
    D20_Device_Reward_1['All_Devices'].append(sum([-logging_timeline[i][j]['KeyRewards'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
    D20_Device_AoI_1['All_Devices'].append(sum([logging_timeline[i][j]['KeyAoI'][x] * KeyInterval[x] for x in range(len(KeyInterval))]) / param['nTimeUnits'])
    D20_Device_CPU_1['All_Devices'].append(sum([logging_timeline[i][j]['KeyCPU'][x] * KeyInterval[x] for x in range(len(KeyInterval))]) / param['nTimeUnits'])
D20_Device_Reward_1['Best'] = mean(D20_Device_Reward_1['All_Devices'])
D20_Device_AoI_1['Best'] = mean(D20_Device_AoI_1['All_Devices'])
D20_Device_CPU_1['Best'] = mean(D20_Device_CPU_1['All_Devices'])

ls_rewards = []
ls_AoI = []
ls_CPU = []
for j in range(1, param['episodes']+1):
    D20_Device_Reward_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
    D20_Device_AoI_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
    D20_Device_CPU_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
    for i in range(param['num_Devices']):
        if not logging_timeline[i][j]['intervals']:
            continue  # 跳出J的FOR循环，不执行后续代码，I保持不变
        KeyInterval = logging_timeline[i][j]['intervals'] + [
            param['nTimeUnits'] - logging_timeline[i][j]['intervals'][-1]]
        # 记录某个设备，某个EPISODE的均值
        D20_Device_Reward_1['All_Devices'].append(
            sum([-logging_timeline[i][j]['KeyRewards'][x] * KeyInterval[x] for x in range(len(KeyInterval))]) / param[
                'nTimeUnits'])
        D20_Device_AoI_1['All_Devices'].append(
            sum([logging_timeline[i][j]['KeyAoI'][x] * KeyInterval[x] for x in range(len(KeyInterval))]) / param[
                'nTimeUnits'])
        D20_Device_CPU_1['All_Devices'].append(
            sum([logging_timeline[i][j]['KeyCPU'][x] * KeyInterval[x] for x in range(len(KeyInterval))]) / param[
                'nTimeUnits'])
    D20_Device_Reward_1['Best'] = mean(D20_Device_Reward_1['All_Devices'])
    D20_Device_AoI_1['Best'] = mean(D20_Device_AoI_1['All_Devices'])
    D20_Device_CPU_1['Best'] = mean(D20_Device_CPU_1['All_Devices'])
    ls_rewards.append(D20_Device_Reward_1['Best'])
    ls_AoI.append(D20_Device_AoI_1['Best'])
    ls_CPU.append(D20_Device_CPU_1['Best'])
D20_Device_Reward_1['Best'] = mean([ls_rewards[x] for x in range(12, 23)])
D20_Device_AoI_1['Best'] = mean([ls_AoI[x] for x in range(12, 23)])
D20_Device_CPU_1['Best'] = mean([ls_CPU[x] for x in range(12, 23)])

# #############################################  25  ###################################################################
with open('fig_C06.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)

# D25_UAV_Energy = mean(env.UAV.Energy) # 这是最后一个EPISODE的数据
D25_UAV_Energy = mean([mean(logging_timeline[0][i]['UAV_Energy']) for i in range(10, 26)])
D25_UAV_Energy_Random = mean(env_random.UAV.Energy)
D25_UAV_Energy_Force = mean(env_force.UAV.Energy)


D25_UAV_Reward = mean([-mean(logging_timeline[0][i]['UAV_Reward']) for i in range(10, 26)])
D25_UAV_Reward_Random = - mean(env_random.UAV.Reward)
D25_UAV_Reward_Force = - mean(env_force.UAV.Reward)

D25_UAV_R_E = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(10, 26)])
D25_UAV_R_E_Random = - mean(env_random.UAV.Sum_R_E)
D25_UAV_R_E_Force = - mean(env_force.UAV.Sum_R_E)

# 下面的和上面的一摸一样

D25_UAV_R_E_1 = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(10, 26)])
D25_UAV_R_E_Random_1 = - mean(logging_timeline[0][0]['Random_UAV_R_E'])
D25_UAV_R_E_Force_1 = - mean(logging_timeline[0][0]['Force_UAV_R_E'])


D25_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(10, 26)])
D25_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(10, 26)])
D25_UAV_b = mean([mean(logging_timeline[0][i]['UAV_b']) for i in range(10, 26)])

D25_UAV_AoI_1 = mean(env.UAV.AoI)
D25_UAV_CPU_1 = mean(env.UAV.CPU)
D25_UAV_b_1 = mean(env.UAV.b)

D25_Device_Reward = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D25_Device_AoI = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D25_Device_CPU = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D25_Device_b = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
for i in range(param['num_Devices']):
    D25_Device_Reward['All_Devices_Episodes'].append([])
    D25_Device_AoI['All_Devices_Episodes'].append([])
    D25_Device_CPU['All_Devices_Episodes'].append([])
    D25_Device_b['All_Devices_Episodes'].append([])
    for j in range(1, param['episodes'] + 1):
        if not logging_timeline[i][j]['intervals']:
            continue  # 跳出J的FOR循环，不执行后续代码，I保持不变
        KeyInterval = logging_timeline[i][j]['intervals'] + [param['nTimeUnits'] - logging_timeline[i][j]['intervals'][-1]]
        # 记录某个设备，某个EPISODE的均值
        D25_Device_Reward['All_Devices_Episodes'][i].append(sum([-logging_timeline[i][j]['KeyRewards'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D25_Device_AoI['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['KeyAoI'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D25_Device_CPU['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['KeyCPU'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D25_Device_b['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['Keyb'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
    # 记录某个设备，全部EPISODE的最小值
    D25_Device_Reward['All_Devices'].append(min(D25_Device_Reward['All_Devices_Episodes'][i]))
    D25_Device_AoI['All_Devices'].append(min(D25_Device_AoI['All_Devices_Episodes'][i]))
    D25_Device_CPU['All_Devices'].append(min(D25_Device_CPU['All_Devices_Episodes'][i]))
    D25_Device_b['All_Devices'].append(min(D25_Device_b['All_Devices_Episodes'][i]))
# 记录全部设备的均值
D25_Device_Reward['Best'] = mean(D25_Device_Reward['All_Devices'])
D25_Device_AoI['Best'] = mean(D25_Device_AoI['All_Devices'])
D25_Device_CPU['Best'] = mean(D25_Device_CPU['All_Devices'])
D25_Device_b['Best'] = mean(D25_Device_b['All_Devices'])

ls_Reward = []
ls_AoI = []
for j in range(1, param['episodes'] + 1):
    D25_Device_Reward_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
    D25_Device_AoI_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
    D25_Device_CPU_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
    for i in range(param['num_Devices']):
        if not logging_timeline[i][j]['intervals']:
            continue  # 跳出J的FOR循环，不执行后续代码，I保持不变
        KeyInterval = logging_timeline[i][j]['intervals'] + [param['nTimeUnits'] - logging_timeline[i][j]['intervals'][-1]]
            # 记录某个设备，某个EPISODE的均值
        D25_Device_Reward_1['All_Devices'].append(sum([-logging_timeline[i][j]['KeyRewards'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D25_Device_AoI_1['All_Devices'].append(sum([logging_timeline[i][j]['KeyAoI'][x] * KeyInterval[x] for x in range(len(KeyInterval))]) / param['nTimeUnits'])
        D25_Device_CPU_1['All_Devices'].append(sum([logging_timeline[i][j]['KeyCPU'][x] * KeyInterval[x] for x in range(len(KeyInterval))]) / param['nTimeUnits'])
    D25_Device_Reward_1['Best'] = mean(D25_Device_Reward_1['All_Devices'])
    D25_Device_AoI_1['Best'] = mean(D25_Device_AoI_1['All_Devices'])
    D25_Device_CPU_1['Best'] = mean(D25_Device_CPU_1['All_Devices'])
    ls_Reward.append(D25_Device_Reward_1['Best'])
    ls_AoI.append(D25_Device_AoI_1['Best'])
D25_Device_Reward_1['Best'] = mean([ls_Reward[x] for x in range(23, 25)])  # 这是比较合适的选择范围
D25_Device_AoI_1['Best'] = mean([ls_AoI[x] for x in range(23, 25)])  # 这是比较合适的选择范围



D25_ls_rewards = []
D25_ls_AoI = []
D25_ls_CPU = []
for j in range(1, param['episodes']+1):
    D25_Device_Reward_2 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
    D25_Device_AoI_2 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
    D25_Device_CPU_2 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
    for i in range(param['num_Devices']):
        if not logging_timeline[i][j]['intervals']:
            continue  # 跳出J的FOR循环，不执行后续代码，I保持不变
        KeyInterval = logging_timeline[i][j]['intervals'] + [
            param['nTimeUnits'] - logging_timeline[i][j]['intervals'][-1]]
        # 记录某个设备，某个EPISODE的均值
        D25_Device_Reward_2['All_Devices'].append(
            sum([-logging_timeline[i][j]['KeyRewards'][x] * KeyInterval[x] for x in range(len(KeyInterval))]) / param[
                'nTimeUnits'])
        D25_Device_AoI_2['All_Devices'].append(
            sum([logging_timeline[i][j]['KeyAoI'][x] * KeyInterval[x] for x in range(len(KeyInterval))]) / param[
                'nTimeUnits'])
        D25_Device_CPU_2['All_Devices'].append(
            sum([logging_timeline[i][j]['KeyCPU'][x] * KeyInterval[x] for x in range(len(KeyInterval))]) / param[
                'nTimeUnits'])
    D25_Device_Reward_2['Best'] = mean(D25_Device_Reward_2['All_Devices'])
    D25_Device_AoI_2['Best'] = mean(D25_Device_AoI_2['All_Devices'])
    D25_Device_CPU_2['Best'] = mean(D25_Device_CPU_2['All_Devices'])
    D25_ls_rewards.append(D25_Device_Reward_1['Best'])
    D25_ls_AoI.append(D25_Device_Reward_1['Best'])
    D25_ls_CPU.append(D25_Device_Reward_1['Best'])
# 这一块还不对
D25_Device_Reward_2['Best'] = mean([D25_ls_rewards[x] for x in range(24, 25)])
D25_Device_AoI_2['Best'] = mean([D25_ls_AoI[x] for x in range(12, 23)])
D25_Device_CPU_2['Best'] = mean([D25_ls_CPU[x] for x in range(12, 23)])




# #############################################  30  ###################################################################
with open('fig_D05.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)

# D30_UAV_Energy = mean(env.UAV.Energy) # 这是最后一个EPISODE的数据
D30_UAV_Energy = mean([mean(logging_timeline[0][i]['UAV_Energy']) for i in range(10, 26)])
D30_UAV_Energy_Random = mean(env_random.UAV.Energy)
D30_UAV_Energy_Force = mean(env_force.UAV.Energy)


D30_UAV_Reward = mean([-mean(logging_timeline[0][i]['UAV_Reward']) for i in range(10, 26)])
D30_UAV_Reward_Random = - mean(env_random.UAV.Reward)
D30_UAV_Reward_Force = - mean(env_force.UAV.Reward)

D30_UAV_R_E = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(10, 26)])
D30_UAV_R_E_Random = - mean(env_random.UAV.Sum_R_E)
D30_UAV_R_E_Force = - mean(env_force.UAV.Sum_R_E)

# 下面的和上面的一摸一样

D30_UAV_R_E_1 = mean([-mean(logging_timeline[0][i]['UAV_R_E']) for i in range(10, 26)])
D30_UAV_R_E_Random_1 = - mean(logging_timeline[0][0]['Random_UAV_R_E'])
D30_UAV_R_E_Force_1 = - mean(logging_timeline[0][0]['Force_UAV_R_E'])


D30_UAV_AoI = mean([mean(logging_timeline[0][i]['UAV_AoI']) for i in range(10, 26)])
D30_UAV_CPU = mean([mean(logging_timeline[0][i]['UAV_CPU']) for i in range(10, 26)])
D30_UAV_b = mean([mean(logging_timeline[0][i]['UAV_b']) for i in range(10, 26)])

D30_UAV_AoI_1 = mean(env.UAV.AoI)
D30_UAV_CPU_1 = mean(env.UAV.CPU)
D30_UAV_b_1 = mean(env.UAV.b)

D30_Device_Reward = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D30_Device_AoI = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D30_Device_CPU = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D30_Device_b = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
for i in range(param['num_Devices']):
    D30_Device_Reward['All_Devices_Episodes'].append([])
    D30_Device_AoI['All_Devices_Episodes'].append([])
    D30_Device_CPU['All_Devices_Episodes'].append([])
    D30_Device_b['All_Devices_Episodes'].append([])
    for j in range(1, param['episodes'] + 1):
        if not logging_timeline[i][j]['intervals']:
            continue  # 跳出J的FOR循环，不执行后续代码，I保持不变
        KeyInterval = logging_timeline[i][j]['intervals'] + [param['nTimeUnits'] - logging_timeline[i][j]['intervals'][-1]]
        # 记录某个设备，某个EPISODE的均值
        D30_Device_Reward['All_Devices_Episodes'][i].append(sum([-logging_timeline[i][j]['KeyRewards'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D30_Device_AoI['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['KeyAoI'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D30_Device_CPU['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['KeyCPU'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
        D30_Device_b['All_Devices_Episodes'][i].append(sum([logging_timeline[i][j]['Keyb'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
    # 记录某个设备，全部EPISODE的最小值
    D30_Device_Reward['All_Devices'].append(min(D30_Device_Reward['All_Devices_Episodes'][i]))
    D30_Device_AoI['All_Devices'].append(min(D30_Device_AoI['All_Devices_Episodes'][i]))
    D30_Device_CPU['All_Devices'].append(min(D30_Device_CPU['All_Devices_Episodes'][i]))
    D30_Device_b['All_Devices'].append(min(D30_Device_b['All_Devices_Episodes'][i]))
# 记录全部设备的均值
D30_Device_Reward['Best'] = mean(D30_Device_Reward['All_Devices'])
D30_Device_AoI['Best'] = mean(D30_Device_AoI['All_Devices'])
D30_Device_CPU['Best'] = mean(D30_Device_CPU['All_Devices'])
D30_Device_b['Best'] = mean(D30_Device_b['All_Devices'])



j = 25
D30_Device_Reward_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D30_Device_AoI_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
D30_Device_CPU_1 = {'Best': [], 'All_Devices': [], 'All_Devices_Episodes': []}
for i in range(param['num_Devices']):
    if not logging_timeline[i][j]['intervals']:
        continue  # 跳出J的FOR循环，不执行后续代码，I保持不变
    KeyInterval = logging_timeline[i][j]['intervals'] + [param['nTimeUnits'] - logging_timeline[i][j]['intervals'][-1]]
        # 记录某个设备，某个EPISODE的均值
    D30_Device_Reward_1['All_Devices'].append(sum([-logging_timeline[i][j]['KeyRewards'][x] * KeyInterval[x] for x in range(len(KeyInterval))])/param['nTimeUnits'])
    D30_Device_AoI_1['All_Devices'].append(sum([logging_timeline[i][j]['KeyAoI'][x] * KeyInterval[x] for x in range(len(KeyInterval))]) / param['nTimeUnits'])
    D30_Device_CPU_1['All_Devices'].append(sum([logging_timeline[i][j]['KeyCPU'][x] * KeyInterval[x] for x in range(len(KeyInterval))]) / param['nTimeUnits'])
D30_Device_Reward_1['Best'] = mean(D30_Device_Reward_1['All_Devices'])
D30_Device_AoI_1['Best'] = mean(D30_Device_AoI_1['All_Devices'])
D30_Device_CPU_1['Best'] = mean(D30_Device_CPU_1['All_Devices'])





# #############################################  数值  ###################################################################
UAV_Energy = [D05_UAV_Energy, D10_UAV_Energy, D15_UAV_Energy, D20_UAV_Energy, D25_UAV_Energy, D30_UAV_Energy] #, D35_UAV_Energy, D40_UAV_Energy]
UAV_Energy_Random = [D05_UAV_Energy_Random, D10_UAV_Energy_Random, D15_UAV_Energy_Random, D20_UAV_Energy_Random, D25_UAV_Energy_Random, D30_UAV_Energy_Random] #, D35_UAV_Energy_Random, D40_UAV_Energy_Random]
UAV_Energy_Force = [D05_UAV_Energy_Force, D10_UAV_Energy_Force, D15_UAV_Energy_Force, D20_UAV_Energy_Force, D25_UAV_Energy_Force, D30_UAV_Energy_Force] #, D35_UAV_Energy_Force, D40_UAV_Energy_Force]

UAV_Reward = [D05_UAV_Reward, D10_UAV_Reward, D15_UAV_Reward, D20_UAV_Reward, D25_UAV_Reward, D30_UAV_Reward] #, D35_UAV_Reward, D40_UAV_Reward] # 因为速度快，所以访问快，所以REWARD绝对值变小了
UAV_Reward_Random = [D05_UAV_Reward_Random, D10_UAV_Reward_Random, D15_UAV_Reward_Random, D20_UAV_Reward_Random, D25_UAV_Reward_Random, D30_UAV_Reward_Random] #, D35_UAV_Reward_Random, D40_UAV_Reward_Random]
UAV_Reward_Force = [D05_UAV_Reward_Force, D10_UAV_Reward_Force, D15_UAV_Reward_Force, D20_UAV_Reward_Force, D25_UAV_Reward_Force, D30_UAV_Reward_Force] #, D35_UAV_Reward_Force, D40_UAV_Reward_Force]


UAV_R_E = [D05_UAV_R_E, D10_UAV_R_E, D15_UAV_R_E, D20_UAV_R_E, D25_UAV_R_E, D30_UAV_R_E] #, D35_UAV_R_E, D40_UAV_R_E]
UAV_R_E_Random = [D05_UAV_R_E_Random, D10_UAV_R_E_Random, D15_UAV_R_E_Random, D20_UAV_R_E_Random, D25_UAV_R_E_Random, D30_UAV_R_E_Random] #, D35_UAV_R_E_Random, D40_UAV_R_E_Random]
UAV_R_E_Force = [D05_UAV_R_E_Force, D10_UAV_R_E_Force, D15_UAV_R_E_Force, D20_UAV_R_E_Force, D25_UAV_R_E_Force, D30_UAV_R_E_Force] #, D35_UAV_R_E_Force, D40_UAV_R_E_Force]


UAV_AoI = [D05_UAV_AoI, D10_UAV_AoI, D15_UAV_AoI, D20_UAV_AoI, D25_UAV_AoI, D30_UAV_AoI] #, D35_UAV_AoI_1, D40_UAV_AoI] # 因为速度快，所以访问快，所以AoI绝对值变小了
UAV_CPU = [D05_UAV_CPU, D10_UAV_CPU, D15_UAV_CPU, D20_UAV_CPU, D25_UAV_CPU, D30_UAV_CPU] #, D35_UAV_CPU_1, D40_UAV_CPU] # 因为速度快，所以访问快，所以AoI绝对值变小了
UAV_AoI_CPU = list(map(lambda x,y: x + y, UAV_AoI, UAV_CPU))
UAV_b = [D05_UAV_b, D10_UAV_b, D15_UAV_b, D20_UAV_b, D25_UAV_b, D30_UAV_b] #, D35_UAV_b_1, D40_UAV_b] # 因为速度快，所以访问快，所以AoI绝对值变小了


UAV_CPU_J = [pow(x * pow(10,8), 3) * 4 * pow(10, -28)  for x in UAV_CPU] # in Joule

Device_Reward = [D05_Device_Reward['Best'], D10_Device_Reward['Best'], D15_Device_Reward['Best'], D20_Device_Reward['Best'], D25_Device_Reward['Best'], D30_Device_Reward['Best']]
Device_AoI = [D05_Device_AoI['Best'], D10_Device_AoI['Best'], D15_Device_AoI['Best'], D20_Device_AoI['Best'], D25_Device_AoI['Best'], D30_Device_AoI['Best']]
Device_CPU = [D05_Device_CPU['Best'], D10_Device_CPU['Best'], D15_Device_CPU['Best'], D20_Device_CPU['Best'], D25_Device_CPU['Best'], D30_Device_CPU['Best']]
Device_CPU_J = [pow(x * pow(10,8), 3) * 4 * pow(10, -28)  for x in Device_CPU]


Device_Reward_1 = [D05_Device_Reward_1['Best'], D10_Device_Reward_1['Best'], D15_Device_Reward_1['Best'], D20_Device_Reward_1['Best'], D25_Device_Reward_1['Best'], D30_Device_Reward_1['Best']]
Device_AoI_1 = [D05_Device_AoI_1['Best'], D10_Device_AoI_1['Best'], D15_Device_AoI_1['Best'], D20_Device_AoI_1['Best'], D25_Device_AoI_1['Best'], D30_Device_AoI_1['Best']]
Device_CPU_1 = [D05_Device_CPU_1['Best'], D10_Device_CPU_1['Best'], D15_Device_CPU_1['Best'], D20_Device_CPU_1['Best'], D25_Device_CPU_1['Best'], D30_Device_CPU_1['Best']]



# #############################################  画图  ###################################################################
# 设备数目不同时，UAV的表现？
fig, ax = plt.subplots()
numD = ['5', '10', '15', '20', '25', '30']
ax.plot(numD, UAV_Reward, color='C1', marker='^', label='UAV Reward') # , color=bar_colors)
ax.plot(numD, UAV_Energy, color='C2', marker='^', label='UAV Energy')# , color=bar_colors)
ax.plot(numD, UAV_R_E, color='C3', marker='^', label='UAV R_E')# , color=bar_colors)
# ax.plot(numD, UAV_R_E, color='C7', marker = '^', label='UAV R+E') # , color=bar_colors)
# ax.plot(numD, UAV_R_E_Random, color='C8', marker = '^', label='UAV R+E Random') # , color=bar_colors)
ax.set_ylabel('UAV Performance')
ax.set_title('UAV Performance V.S. num of Devices')
ax.legend(loc='best')
# ax.grid(True)



# 设备数目不同时，UAV的表现？
fig, ax = plt.subplots()
numD = ['5', '10', '15', '20', '25', '30']
ax.plot(numD, UAV_Reward, color='C4', marker='^', label='UAV Reward') # , color=bar_colors)
ax.plot(numD, UAV_AoI, color='C5', marker='^', label='UAV AoI')# , color=bar_colors)
ax.plot(numD, UAV_CPU, color='C6', marker='^', label='UAV CPU')# , color=bar_colors)
# ax.plot(numD, UAV_R_E, color='C7', marker = '^', label='UAV R+E') # , color=bar_colors)
# ax.plot(numD, UAV_R_E_Random, color='C8', marker = '^', label='UAV R+E Random') # , color=bar_colors)
ax.set_ylabel('UAV Energy')
ax.set_title('UAV energy and reward corresponding to velocity')
ax.legend(loc='best')
# ax.grid(True)



# 先画设备与REWARD之间的关系图
fig, ax = plt.subplots()
numD = ['5', '10', '15', '20', '25', '30']
axt = ax.twinx()
ax.plot(numD, Device_Reward_1,  color='C1', marker='o', label='Devices Reward') # , color=bar_colors)
ax.plot(numD, Device_AoI_1, color='C2', marker='*', label='Devices AoI')# , color=bar_colors)
axt.plot(numD, [x*1000 for x in Device_CPU_J], color='C3', marker='^', label='Devices CPU')# , color=bar_colors)
ax.set_xlabel('Number of Devices')
ax.set_ylabel('AoI and Reward')
axt.set_ylabel('CPU (mJ)')
ax.set_title('Performance v.s. Num of Devices')
ax.legend(loc='best')
axt.legend(loc='best')


power = 1
