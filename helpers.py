import numpy as np


def normalize_1_1(x, min, max):
    return np.array([((2 * ((x[0]-(min))/(max-(min)))) - 1), ((2 * ((x[1]-(min))/(max-(min)))) - 1)])


def normalize_0_1(x, min, max):
    return np.array([(x[0]-(min))/(max-(min)), (x[1]-(min))/(max-(min))])


def interval_map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def normalize(x, x_min, x_max, a=0.0, b=1.0):
    return interval_map(x, x_min, x_max, a, b)


def scale(x, a, b):
    return interval_map(x, 0.0, 1.0, a, b)


def sensors_offset(distance, minDetection, noDetection):
    return (1 - ((distance - minDetection) / (noDetection - minDetection)))


def f_wheel_center(wheels, min, max):
    return normalize((((wheels[0]) + (wheels[1])) / 2), min, max)


def f_straight_movements(wheels, min, max):
    return (1 - (np.sqrt(normalize(np.absolute(wheels[0] - wheels[1]), min, max))))


def f_obstacle_dist(sensors):
    return (1 - np.amax(sensors))
