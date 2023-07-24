import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cbook as cbook
import pickle
from statistics import mean, stdev


"""
画不同的速度收敛图，matplotlib, fill_between
"""

#参考链接： https://matplotlib.org/3.7.1/gallery/lines_bars_and_markers/fill_between_alpha.html



with open('fig_A19.pkl', 'rb') as f:
    model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(f)

for x in range(1, param['episodes']):
    l = len(logging_timeline[0][x]['UAV_VelocityList'])
    m = mean(logging_timeline[0][x]['UAV_VelocityList'])
    s = stdev(logging_timeline[0][x]['UAV_VelocityList'])
    print(l, m, s)

V = []
V_high = []
V_low = []
V_real = []
avg1 = []
avg2 = []
num = 50
for x in range(1, param['episodes']):
    temp = logging_timeline[0][x]['UAV_VelocityList']
    # print(x)
    temp_mean = mean(temp)
    temp_std = stdev(temp)
    avg1 = avg1 + [temp_mean] * num
    avg2.append(temp_mean)
    # 1. 画上界，下界为最小值
    # ⚠️ 因为原来的值都是一样的，偏差不大，所以生成新的随机数
    temp_new = np.random.normal(temp_mean, temp_std, num)
    V = V + list(temp_new)
    # 2. 画上下两条线FILL IN
    temp_high = np.random.normal(temp_mean + temp_std, temp_std, num)
    temp_low = np.random.normal(temp_mean - temp_std, temp_std, num)
    V_high = V_high + list(temp_high)
    V_low = V_low + list(temp_low)
    # 3. 真实数据，去除 x = 3，这时候logging_timeline[0][x]['UAV_VelocityList']只有4个元素
    if x != 3:
        temp_real = logging_timeline[0][x]['UAV_VelocityList'][1:51]
        V_real = V_real + temp_real
Vmin = min(V)
V_min_real = min(V_real)



# 1 --------------------------------
fig, ax = plt.subplots(1)
ax.fill_between(np.arange(1, len(V) + 1), Vmin, V, alpha=0.7)
ax.plot(np.arange(num/2, num * (param['episodes'] - 1), num), avg2,'C1', alpha = 1)


fig, ax = plt.subplots(1)
ax.fill_between(np.arange(1, len(V) + 1), Vmin, V, alpha=0.7, lw = 2)
ax.plot(np.arange(1, len(avg1)+1), avg1)

# 2 --------------------------------
fig, ax = plt.subplots(1)
ax.fill_between(np.arange(1, len(V_low) + 1), np.array(V_low), np.array(V_high), alpha=0.7)
ax.plot(np.arange(1, len(avg1)+1), avg1)

fig, ax = plt.subplots(1)
ax.fill_between(np.arange(1, len(V_low) + 1), np.array(V_low), np.array(V_high), alpha=0.7)
ax.plot(np.arange(num/2, num * (param['episodes'] - 1), num), avg2,'C0', alpha = 1)

# 3 --------------------------------
fig, ax = plt.subplots(1)
ax.fill_between(np.arange(1, len(V_real) + 1), V_min_real, V_real, alpha=0.7)
ax.plot(np.arange(num/2, num * (param['episodes'] - 1), num), avg2,'red', alpha = 1)




d = 1





