import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pickle
from associations import pg_rl
from UAV_Device import Device
from UAV_Device import UAV_Obj


def main():
    """
    This function is used to simulate the flying of UAV over Devices
    The main steps are as follows:
    Step 1: Decide the
    :return:
    """
    # Prepare the model and parameters
    with open('mu_sig.pkl', 'rb') as f:
        mu, sig = pickle.load(f)  # Load the mu and sig to extract normalized feature

    with open('TaDeLL_result_k_2.pkl', 'rb') as f:
        means_pg, means_tadell, niter, TaDeLL_Model, tasks0, tasks, testing_tasks, testing_tasks_pg, testing_tasks_TaDeLL = pickle.load(f)


    nTimeUnits = 300
    num_Devices = 2  # Number of Devices in the field
    nTasks = 3  # Number of tasks for each device
    V = 20  # m/s  the velocity of the UAV
    edge = 200 # The edge length of the square field (unit: m)
    dist = V * 2  # The minimum distance between devices

    param = {'nTimeUnits':nTimeUnits, 'num_Devices':num_Devices, 'nTasks':nTasks, 'V':V, 'edge':edge, 'dist':dist}
    # Unpacking the parameter dictionary
    # nTimeUnits = param['nTimeUnits']
    # num_Devices = param['num_Devices']
    # nTasks = param['nTasks']
    # V = param['V']
    # edge = param['edge']
    # dist = param['dist']

    UAV = UAV_Obj(V)  # One UAV and multiple Devices
    Devices = gen_Devices(mu, sig, **param)

    PolType = "Random"
    Devices_Random = copy.deepcopy(Devices)
    UAV_Random = copy.deepcopy(UAV)
    Devices_Random = UAV_Random.UAV_Fly(Devices_Random, TaDeLL_Model, PolType, **param)
    Rewards_Random, Rewards_Random_Natural = UAV_Random.compRewards(Devices_Random)  # Compute reward using device.KeyTime and device.KeyPol

    # PolType = "LSTM"
    # Devices_LSTM = copy.deepcopy(Devices)
    # UAV_LSTM = copy.deepcopy(UAV)
    # Devices_LSTM = UAV_LSTM.UAV_Fly(Devices_LSTM, TaDeLL_Model, PolType, **param)
    # Rewards_LSTM = UAV_LSTM.compRewards(Devices_LSTM)

    # Update the profit_list

    # Plotting phase
    plt.ion()
    fig, ax = plt.subplots(2,num_Devices, sharex=True)
    fig.suptitle('Learning Process')
    for i in range(num_Devices):
        ax[0, i].plot(np.arange(nTimeUnits), Rewards_Random[i], label="Warm Start")
        ax[0, i].plot(np.arange(nTimeUnits), Rewards_Random_Natural[i], label="Natural")
        # ax[i].set_title('device')
        ax[0, i].legend()
    plt.show()


    # ax[0].plot(np.arange(nTimeUnits), Rewards_Random[0], label='device 0')
    # ax[1].plot(np.arange(nTimeUnits), Rewards_Random[1], label='device 1')
    # # Replacement
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    # ax1.plot()
    # ax1.set_title()
    # ax1.set_ylabel()
    # ax1.setylin()
    # plt.show()

    print("UAV is generated successfully!")


def too_close(device, Devices, dist):
    flag = False
    for i in range(len(Devices)):
        distance = np.linalg.norm(device.location - Devices[i].location)
        if distance < dist:
            flag = True
            break
    return  flag

def gen_Devices(mu, sig, **param):
    # Unpacking the parameter dictionary
    nTimeUnits = param['nTimeUnits']
    num_Devices = param['num_Devices']
    nTasks = param['nTasks']
    edge = param['edge']
    dist = param['dist']

    # Generate the list of Devices
    # FIXME: Generate and filter v.s. generate randomly
    Devices = []  # The list of all the devices
    for i in range(num_Devices):
        flag = True
        while flag:
            device = Device(mu, sig, nTimeUnits, nTasks, edge)
            if not too_close(device, Devices, dist):
                flag = False
        Devices.append(Device(mu, sig, nTimeUnits, nTasks, edge))
    return Devices


if __name__ == '__main__':
    main()