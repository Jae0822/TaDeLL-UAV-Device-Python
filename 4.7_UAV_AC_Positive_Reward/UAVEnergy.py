import numpy as np
import math
import matplotlib.pyplot as plt

def UAV_Energy(V):
    W = 20
    rho = 1.225
    R = 0.4
    A = 0.503
    Omega = 200
    U_tip = 120
    b = 4
    c = 0.0157
    s = 0.05
    S_FP = 0.0151
    d0 = 0.6
    k = 0.1
    v0 = 4.03
    delta = 0.012

    P0 = (1/8) * delta * rho * s * A * pow(Omega, 3) * pow(R, 3)

    Pi = (1 + k) * pow(W, 3/2) / math.sqrt(2 * rho * A)

    PV = P0 * (1 + 3 * V ** 2 / U_tip ** 2) + Pi * pow(math.sqrt(1 + V ** 4 / (4 * v0 ** 4)) - V ** 2 / (2 * v0 ** 2),
                                                       1 / 2) + (1 / 2) * d0 * rho * s * A * V ** 3
    return PV

# P = []
# for V in range(1, 30):
#     PV = P0 * (1 + 3 * V ** 2 / U_tip ** 2) + Pi * pow(math.sqrt(1 + V ** 4 / (4 * v0 ** 4)) - V ** 2 / (2 * v0 ** 2),
#                                                        1 / 2) + (1 / 2) * d0 * rho * s * A * V ** 3
#     P.append(PV)
# fig, ax = plt.subplots(1)
# ax.plot(np.arange(1, 30), P)
# ax.set_xlabel('UAV Speed V (m/s)')
# ax.set_ylabel('Required Power (W)')
# ax.grid()
# ax.set_title('Propulsion power consumption versus speed V for rotary-wing UAV')
# plt.show()