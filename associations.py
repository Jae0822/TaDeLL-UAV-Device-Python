import numpy as np

def pg_rl(task, niter=50, lr=0.08):
    """
    Basic policy gradient learning can work fine for env.Task
    """
    # niter = 50
    values = []
    for i in range(niter):  # Iterate policy gradient process
        path = task.collect_path(task.policy["theta"])

        djd_theta = task.djd_nac(path)
        task.policy["theta"] = task.policy["theta"] + lr * djd_theta

        values.append(task.get_value(task.policy["theta"]))

        # print(i)
        # print("djd_theta", djd_theta, "policy:", task.policy["theta"], "rewards:", values[-1])
    values_array = np.array(values)

    # print("rewards_array:", values_array)
    # # Plotting procedure
    # plt.ion()
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(niter) + 1, values_array)
    # fig.show()
    # plt.ioff()
    # print("Hello")

    return values_array




def pg_rl_values_policies_gradients(task, niter=50, lr=0.08):
    """
    Basic policy gradient learning can work fine for env.Task
    """
    # niter = 50
    values = []
    policies = []
    gradients = []

    values.append(task.get_value(task.policy["theta"]))
    policies.append(task.policy["theta"])

    for i in range(niter):  # Iterate policy gradient process
        path = task.collect_path(task.policy["theta"])

        djd_theta = task.djd_nac(path)
        task.policy["theta"] = task.policy["theta"] + lr * djd_theta

        values.append(task.get_value(task.policy["theta"]))
        policies.append(task.policy["theta"])
        gradients.append(djd_theta)

        # print(i)
        # print("djd_theta", djd_theta, "policy:", task.policy["theta"], "rewards:", values[-1])
    values_array = np.array(values)
    policies_array = np.array(policies)
    gradients_array = np.array(gradients)

    # print("rewards_array:", values_array)
    # # Plotting procedure
    # plt.ion()
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(niter) + 1, values_array)
    # fig.show()
    # plt.ioff()
    # print("Hello")

    return values_array, policies_array, gradients_array