from __future__ import print_function
import os
import neat
import vrep
import numpy as np
from datetime import datetime, timedelta
import time
import uuid
from robot import EvolvedRobot
from helpers import f_wheel_center, f_straight_movements, f_pain, scale
import math
from argparse import ArgumentParser
import configparser
import settings
from functools import partial
from neat import ThreadedEvaluator
from neat import ParallelEvaluator
import yaml
from functools import partial
try:
    # pylint: disable=import-error
    import Queue as queue
except ImportError:
    # pylint: disable=import-error
    import queue

try:
    import threading
except ImportError:  # pragma: no cover
    import dummy_threading as threading
    HAVE_THREADS = False
else:
    HAVE_THREADS = True

try:
    # pylint: disable=import-error
    import Queue as queue
except ImportError:
    # pylint: disable=import-error
    import queue

import warnings
import pdb
from multiprocessing import Pool, Queue

settings.init()


class ThreadedEvaluatorCustom(object):
    """
    A threaded genome evaluator.
    Useful on python implementations without GIL (Global Interpreter Lock).
    """

    def __init__(self, clients, num_workers, eval_function):
        """
        eval_function should take two arguments (a genome object and the
        configuration) and return a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.ports = clients
        self.workers = []
        self.working = False
        self.inqueue = queue.Queue()
        self.outqueue = queue.Queue()

        if not HAVE_THREADS:  # pragma: no cover
            warnings.warn(
                "No threads available; use ParallelEvaluator, not ThreadedEvaluator")

    def __del__(self):
        """
        Called on deletion of the object. We stop our workers here.
        WARNING: __del__ may not always work!
        Please stop the threads explicitly by calling self.stop()!
        TODO: ensure that there are no reference-cycles.
        """
        if self.working:
            self.stop()

    def start(self):
        """Starts the worker threads"""
        if self.working:
            return
        self.working = True
        for i in range(self.num_workers):
            w = threading.Thread(
                name="Worker Thread #{i}".format(i=i),
                target=self._worker,
                args=(self.ports[i],),
            )
            w.daemon = True
            w.start()
            print("thread_id = {0} client_id = {1}".format(
                w.getName(), self.ports[i]))
            self.workers.append(w)

    def stop(self):
        """Stops the worker threads and waits for them to finish"""
        self.working = False
        for w in self.workers:
            w.join()
        self.workers = []

    def _worker(self, client_id):
        """The worker function"""
        while self.working:
            try:
                genome_id, genome, config = self.inqueue.get(
                    block=True,
                    timeout=0.2,
                )
            except queue.Empty:
                continue
            f = self.eval_function(client_id, genome, config)
            self.outqueue.put((genome_id, genome, f))

    def evaluate(self, genomes, config):
        """Evaluate the genomes"""
        if not self.working:
            self.start()
        p = 0
        for genome_id, genome in genomes:
            p += 1
            self.inqueue.put((genome_id, genome, config))

        # assign the fitness back to each genome
        while p > 0:
            p -= 1
            ignored_genome_id, genome, fitness = self.outqueue.get()
            genome.fitness = fitness


def port_initializer(q):
    global client_id
    client_id = q.get()
    print("worked_id = {0} client_id = {1}".format(os.getpid(), client_id))


def vrep_ports():
    with open("vrep_ports.yml", 'r') as f:
        portConfig = yaml.load(f)
    return portConfig['ports']


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def eval_genome(client_id, genome, config):
    print('process id = ', os.getpid())

    robot = EvolvedRobot(
        None,
        client_id=client_id,
        id=None,
        op_mode=settings.OP_MODE)

    # Enable the synchronous mode
    vrep.simxSynchronous(client_id, True)
    if (vrep.simxStartSimulation(client_id, vrep.simx_opmode_oneshot) == -1):
        print(client_id, 'Failed to start the simulation')
        print('Program ended')
        return

    robot.chromosome = genome
    robot.wheel_speeds = np.array([])
    robot.sensor_activation = np.array([])
    robot.norm_wheel_speeds = np.array([])
    individual = robot
    id = uuid.uuid1()

    start_position = None
    # collistion detection initialization
    _, collision_handle = vrep.simxGetCollisionHandle(
        client_id, 'robot_collision', vrep.simx_opmode_blocking)
    collision = False

    now = datetime.now()
    fitness_agg = np.array([])
    scaled_output = np.array([])
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    if start_position is None:
        start_position = individual.position

    _, collision = vrep.simxReadCollision(
        client_id, collision_handle, vrep.simx_opmode_streaming)

    while not collision and datetime.now() - now < timedelta(seconds=settings.RUNTIME):

        # The first simulation step waits for a trigger before being executed
        vrep.simxSynchronousTrigger(client_id)

        _, collision = vrep.simxReadCollision(
            client_id, collision_handle, vrep.simx_opmode_buffer)

        individual.neuro_loop()
        print(individual.sensor_activation)
        output = net.activate(individual.sensor_activation)
        # normalize motor wheel wheel_speeds [0.0, 2.0] - robot
        scaled_output = np.array([scale(xi, -2.0, 2.0) for xi in output])

        if settings.DEBUG:
            individual.logger.info('Wheels {}'.format(scaled_output))

        individual.set_motors(*list(scaled_output))

        # After this call, the first simulation step is finished
        vrep.simxGetPingTime(client_id)

        # Fitness function; each feature;
        # V - wheel center
        V = f_wheel_center(output[0], output[1])
        if settings.DEBUG:
            individual.logger.info('f_wheel_center {}'.format(V))

        # pleasure - straight movements
        pleasure = f_straight_movements(output[0], output[1])
        if settings.DEBUG:
            individual.logger.info(
                'f_straight_movements {}'.format(pleasure))

        # pain - closer to an obstacle more pain
        pain = f_pain(individual.sensor_activation)
        if settings.DEBUG:
            individual.logger.info('f_pain {}'.format(pain))

        #  fitness_t at time stamp
        fitness_t = V * pleasure * pain
        fitness_agg = np.append(fitness_agg, fitness_t)

    # behavarioral fitness function
    fitness_bff = [np.sum(fitness_agg)]

    # tailored fitness function
    fitness = fitness_bff[0]  # * fitness_aff[0]

    # Now send some data to V-REP in a non-blocking fashion:
    vrep.simxAddStatusbarMessage(
        client_id, 'fitness: {}'.format(fitness), vrep.simx_opmode_oneshot)

    # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    vrep.simxGetPingTime(client_id)

    print('%s fitness: %f | fitness_bff %f | fitness_aff %f' % (
        str(id), fitness, fitness_bff[0], 0.0))

    if (vrep.simxStopSimulation(client_id, settings.OP_MODE) == -1):
        print('Failed to stop the simulation\n')
        print('Program ended\n')
        return

    time.sleep(1)
    return fitness


def run(config_file):
    print('Neuroevolutionary program started!')
    # Just in case, close all opened connections
    vrep.simxFinish(-1)
    ports = vrep_ports()

    clients = [vrep.simxStart(
        '127.0.0.1',
        port,
        True,
        True,
        5000,
        5) for port in ports]

    print(clients)

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

# Run for up to 300 generations.
    kwargs = {'num_workers': 2, 'eval_function': eval_genome}
    pe = ThreadedEvaluatorCustom(clients, 2, eval_genome)
    winner = p.run(pe.evaluate, 10)
    # save the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.ini')
    run(config_path)
