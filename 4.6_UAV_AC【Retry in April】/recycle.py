parser.add_argument('--nTimeUnits', type=int, default=300,
                    help='number of time units per episode')
parser.add_argument('--nDevices', type=int, default=3,
                    help='number of Devices')
parser.add_argument('--V', type=int, default=20,
                    help='the velocity of the UAV (unit: m/s)')
parser.add_argument('--dist', type=int, default=40,
                    help='The minimum distance between devices')
parser.add_argument('--field', type=int, default=200,
                    help='the edge length of the square field (unit: m)')
parser.add_argument('--interval', type=float, default=[200, 400, 600], nargs='+',
                    help='the new task interval of each device')






















