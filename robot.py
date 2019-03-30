import vrep
import time
import math
from datetime import datetime, timedelta
import numpy as np
import pickle
import logging
from helpers import sensors_offset, normalize
import uuid

PI = math.pi
NUM_SENSORS = 16
PORT_NUM = 19997
RUNTIME = 20
OP_MODE = vrep.simx_opmode_oneshot_wait
X_MIN = 0
X_MAX = 48
DEBUG = False


class Robot:

    def __init__(self, client_id, id, op_mode, noDetection=1.0, minDetection=0.05, initSpeed=2):
        self.id = id
        self.client_id = client_id
        self.op_mode = op_mode

        # Specific props
        self.noDetection = noDetection
        self.minDetection = minDetection
        self.initSpeed = initSpeed
        self.wheel_speeds = np.array([])
        self.sensor_activation = np.array([])
        self.norm_wheel_speeds = np.array([])

        # Custom Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        c_handler = logging.StreamHandler()
        self.logger.setLevel(logging.INFO)
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        self.logger.addHandler(c_handler)

        res, self.body = vrep.simxGetObjectHandle(
            self.client_id, "Pioneer_p3dx%s" %
            self.suffix, self.op_mode)

        # Initialize Motors
        res, self.left_motor = vrep.simxGetObjectHandle(
            self.client_id, "Pioneer_p3dx_leftMotor%s" %
            self.suffix, self.op_mode)
        res, self.right_motor = vrep.simxGetObjectHandle(
            self.client_id, "Pioneer_p3dx_rightMotor%s" %
            self.suffix, self.op_mode)
        self.wheels = [self.left_motor, self.right_motor]

        # Initialize Proximity Sensors
        self.prox_sensors = []
        self.prox_sensors_val = np.array([])
        for i in range(1, NUM_SENSORS + 1):
            res, sensor = vrep.simxGetObjectHandle(
                self.client_id, 'Pioneer_p3dx_ultrasonicSensor%d%s' %
                (i, self.suffix), self.op_mode)
            self.prox_sensors.append(sensor)
            errorCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(
                self.client_id, sensor, vrep.simx_opmode_streaming)
            np.append(self.prox_sensors_val, np.linalg.norm(detectedPoint))

        # Orientation of all the sensors:
        self.sensors_loc = np.array([-PI / 2, -50 / 180.0 * PI, -30 / 180.0 * PI,
                                    -10 / 180.0 * PI, 10 / 180.0 * PI, 30 / 180.0 * PI,
                                     50 / 180.0 * PI, PI / 2, PI / 2, 130 / 180.0 * PI,
                                     150 / 180.0 * PI, 170 / 180.0 * PI, -170 / 180.0 * PI,
                                     -150 / 180.0 * PI, -130 / 180.0 * PI, -PI / 2])

    @property
    def suffix(self):
        if self.id is not None:
            return '#%d' % self.id
        return ''

    def move_forward(self, speed=2.0):
        self.set_motors(speed, speed)

    def move_backward(self, speed=2.0):
        self.set_motors(-speed, -speed)

    def set_motors(self, left, right):
        vrep.simxSetJointTargetVelocity(
            self.client_id,
            self.left_motor,
            left,
            vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(
            self.client_id,
            self.right_motor,
            right,
            vrep.simx_opmode_streaming)

    def set_left_motor(self, left):
        vrep.simxSetJointTargetVelocity(
            self.client_id,
            self.left_motor,
            left,
            vrep.simx_opmode_oneshot)

    def set_right_motor(self, right):
        vrep.simxSetJointTargetVelocity(
            self.client_id,
            self.right_motor,
            right,
            vrep.simx_opmode_oneshot)

    def get_sensor_state(self, sensor):
        errorCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(
            self.client_id, sensor, vrep.simx_opmode_buffer)
        return detectionState

    def get_sensor_distance(self, sensor):
        errorCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(
            self.client_id, sensor, vrep.simx_opmode_buffer)
        return np.linalg.norm(detectedPoint)

    def test_sensors(self):
        while True:
            self.sensor_activation = np.array([])
            for s in self.prox_sensors:
                if self.get_sensor_state(s):
                    # offset
                    activation = sensors_offset(self.get_sensor_distance(s),
                        self.minDetection, self.noDetection)
                    self.sensor_activation = np.append(
                        self.sensor_activation, activation)
                else:
                    self.sensor_activation = np.append(self.sensor_activation, 0)

            self.logger.info('Sensors Activation {}'.format(self.sensor_activation))


    @property
    def position(self):
        returnCode, (x, y, z) = vrep.simxGetObjectPosition(
            self.client_id, self.body, -1, self.op_mode)
        return x, y

    def save_robot(self, filename):
        with open(filename, 'wb') as robot:
            pickle.dump(self, robot)

    def avoid_obstacles(self):
        start_time = datetime.now()

        while datetime.now() - start_time < timedelta(seconds=RUNTIME):
            sensors_val = np.array([])
            for s in self.prox_sensors:
                detectedPoint = self.get_sensor_distance(s)
                sensors_val = np.append(sensors_val, detectedPoint)

            # controller specific - take front sensor values.
            sensor_sq = sensors_val[0:8] * sensors_val[0:8]
            # find sensor where the obstacle is closest
            min_ind = np.where(sensor_sq == np.min(sensor_sq))
            min_ind = min_ind[0][0]

            if sensor_sq[min_ind] < 0.2:
                # sensor which has the obstacle closest to it`
                steer = -1 / self.sensors_loc[min_ind]
            else:
                steer = 0

            v = 1  # forward velocity
            kp = 0.5  # steering gain
            vl = v + kp * steer
            vr = v - kp * steer
            print("V_l = " + str(vl))
            print("V_r = " + str(vr))
            self.set_motors(vl, vr)
            time.sleep(0.2)  # loop executes once every 0.2 seconds (= 5 Hz)

        # Post ALlocation
        errorCode = vrep.simxSetJointTargetVelocity(
            self.client_id, self.left_motor, 0, self.op_mode)
        errorCode = vrep.simxSetJointTargetVelocity(
            self.client_id, self.right_motor, 0, self.op_mode)


class EvolvedRobot(Robot):
    def __init__(self, chromosome, client_id, id, op_mode):
        super().__init__(client_id, id, op_mode)
        self.chromosome = chromosome
        self.unique_id = uuid.uuid1()

    def __str__(self):
        return "Chromosome: %s\n WheelSpeed: %s\n Normalized Speed: %s\n Sensor Activation: %s\n Max Sensor Activation: %s\n" % (
            self.chromosome, self.wheel_speeds, self.norm_wheel_speeds, self.sensor_activation, self.sensor_activation)

    def loop(self):
        wheelspeed = np.array([0.0, 0.0])
        self.sensor_activation = np.array([])

        for i, sensor in enumerate(self.prox_sensors):
            if self.get_sensor_state(sensor):
                # take into account the offset & range; rescale activation
                activation = sensors_offset(self.get_sensor_distance(sensor), self.minDetection, self.noDetection)
                self.sensor_activation = np.append(self.sensor_activation, activation)
                wheelspeed += np.float32(
                    np.array(self.chromosome[i * 4:i * 4 + 2]) * np.array(activation))
            else:
                wheelspeed += np.float32(
                    np.array(self.chromosome[i * 4 + 2:i * 4 + 4]))
                self.sensor_activation = np.append(self.sensor_activation, 0)

        
        if DEBUG: self.logger.info('Wheelspeed aggregation {}'.format(wheelspeed))

        # normalize wheelspeeds in range [0.0, 1.0] - fitness function
        self.norm_wheel_speeds = np.array([normalize(xi, X_MIN, X_MAX, 0.0, 1.0) for xi in wheelspeed])
        if DEBUG: self.logger.info('Normalized WheelSpeed {}'.format(self.norm_wheel_speeds))

        # scale motor wheel_speeds [0.0, 2.0] - robot
        self.wheel_speeds = np.array([ xi * 2.0 for xi in self.norm_wheel_speeds])
        
        if DEBUG: self.logger.info('WheelSpeed {}'.format(self.wheel_speeds))

        self.set_motors(*list(self.wheel_speeds))

    def neuro_loop(self):
        self.sensor_activation = np.array([])
        for i, sensor in enumerate(self.prox_sensors):
            if self.get_sensor_state(sensor):
                activation = sensors_offset(self.get_sensor_distance(sensor), self.minDetection, self.noDetection)
                self.sensor_activation = np.append(
                    self.sensor_activation, activation)
            else:
                self.sensor_activation = np.append(self.sensor_activation, 0)

        if DEBUG: self.logger.info('Sensors Activation {}'.format(self.sensor_activation))

    @property
    def chromosome_size(self):
        return len(self.prox_sensors) * len(self.wheels) * 2


if __name__ == '__main__':
    print('Program started')
    vrep.simxFinish(-1)  # just in case, close all opened connections
    client_id = vrep.simxStart(
        '127.0.0.1',
        PORT_NUM,
        True,
        True,
        5000,
        5)  # Connect to V-REP
    if client_id != -1:
        print('Connected to remote API server')
        op_mode = vrep.simx_opmode_oneshot_wait
        robot = Robot(client_id=client_id, id=None, op_mode=OP_MODE)
        vrep.simxStopSimulation(client_id, op_mode)
        vrep.simxStartSimulation(client_id, op_mode)
        robot.avoid_obstacles()
        vrep.simxStopSimulation(client_id, op_mode)
        vrep.simxFinish(client_id)

        print('Program ended')
    else:
        print('Failed connecting to remote API server')
