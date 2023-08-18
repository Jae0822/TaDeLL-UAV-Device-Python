from IoTEnv import Device

def initialize_fixed_devices(cpu_capacity, param):
    devices = []
    # for i in range(param['num_Devices']):
    #     Devices.append(Device(random.randint(param['freq_low'], param['freq_high']), random.randint(30, 70), param['field']))

    devices.append(Device(530, cpu_capacity, param['field']))
    devices.append(Device(510, cpu_capacity, param['field']))
    devices.append(Device(500, cpu_capacity, param['field']))
    devices.append(Device(485, cpu_capacity, param['field']))
    devices.append(Device(470, cpu_capacity, param['field']))
    devices.append(Device(450, cpu_capacity, param['field']))
    devices.append(Device(430, cpu_capacity, param['field']))
    devices.append(Device(400, cpu_capacity, param['field']))
    devices.append(Device(380, cpu_capacity, param['field']))
    devices.append(Device(350, cpu_capacity, param['field']))
    devices.append(Device(370, cpu_capacity, param['field']))
    devices.append(Device(340, cpu_capacity, param['field']))
    devices.append(Device(330, cpu_capacity, param['field']))
    devices.append(Device(315, cpu_capacity, param['field']))
    devices.append(Device(300, cpu_capacity, param['field']))
    devices.append(Device(275, cpu_capacity, param['field']))
    devices.append(Device(250, cpu_capacity, param['field']))
    devices.append(Device(230, cpu_capacity, param['field']))
    devices.append(Device(215, cpu_capacity, param['field']))
    devices.append(Device(200, cpu_capacity, param['field']))
    devices.append(Device(180, cpu_capacity, param['field']))
    devices.append(Device(150, cpu_capacity, param['field']))
    devices.append(Device(130, cpu_capacity, param['field']))
    devices.append(Device(115, cpu_capacity, param['field']))
    devices.append(Device(100, cpu_capacity, param['field']))
    return devices