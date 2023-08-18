import numpy as np

def pg_rl(task, niter=50, lr=0.08):
    """
    Basic policy gradient learning can work fine for env.Task
    """
    values = []
    for i in range(niter):  # Iterate policy gradient process
        path = task.collect_path(task.policy["theta"])

        djd_theta = task.djd_nac(path)
        task.policy["theta"] = task.policy["theta"] + lr * djd_theta

        values.append(task.get_value(task.policy["theta"]))

    values_array = np.array(values)

    return values_array

