from __future__ import print_function
from helpers import f_wheel_center, \
    f_straight_movements, scale, f_obstacle_dist
from datetime import datetime, timedelta
from neat import ThreadedEvaluator
from robot import EvolvedRobot
from subprocess import Popen
import multiprocessing
import numpy as np
import visualize
import settings
import warnings
import os
import neat
import vrep
import time
import yaml

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

settings.init()


class ParrallelEvolution(object):
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
        self.workers = []
        self.working = False
        self.clients = clients
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
        """Starts the worker threads each connected to specific vrep server"""
        if self.working:
            return
        self.working = True
        for i in range(self.num_workers):
            w = threading.Thread(
                name="Worker Thread #{i}".format(i=i),
                target=self._worker,
                args=(self.clients[i],),
            )
            w.daemon = True
            w.start()
            print("thread_id = {0} client_id = {1}".format(
                w.getName(), self.clients[i]))
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
            f = self.eval_function(client_id, genome, genome_id, config)
            self.inqueue.task_done()
            self.outqueue.put((genome_id, genome, f))

    def evaluate(self, genomes, config):
        """Evaluate the genomes"""
        if not self.working:
            self.start()
        p = 0
        for genome_id, genome in genomes:
            p += 1
            self.inqueue.put((genome_id, genome, config))

        self.inqueue.join()

        # assign the fitness back to each genome
        while p > 0:
            p -= 1
            ignored_genome_id, genome, fitness = self.outqueue.get()
            genome.fitness = fitness


def vrep_ports():
    """Load the vrep ports"""
    with open("vrep_ports.yml", 'r') as f:
        portConfig = yaml.load(f)
    return portConfig['ports']


def eval_genome(client_id, genome, genome_id, config):

    t = threading.currentThread()
    robot = EvolvedRobot(
        genome,
        genome_id,
        client_id=client_id,
        id=None,
        op_mode=settings.OP_MODE)

    # Enable the synchronous mode
    vrep.simxSynchronous(client_id, True)

    if (vrep.simxStartSimulation(client_id, vrep.simx_opmode_oneshot) == -1):
        print(client_id, 'Failed to start the simulation')
        print('Program ended')
        return

    individual = robot
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

        # feed the neural network
        output = net.activate(individual.sensor_activation)

        # normalize motor wheel_speeds [-2.0, 2.0]
        scaled_output = np.array([scale(xi, -2.0, 2.0) for xi in output])

        if settings.DEBUG:
            individual.logger.info('Wheels {}'.format(scaled_output))

        individual.set_motors(*list(scaled_output))

        # After this call, the first simulation step is finished
        vrep.simxGetPingTime(client_id)

        # Fitness function; each feature;
        # V - wheel center
        V = f_wheel_center(scaled_output, -2.0, 2.0)

        if settings.DEBUG:
            individual.logger.info('f_wheel_center {}'.format(V))

        # pleasure - straight movements
        pleasure = f_straight_movements(scaled_output, 0.0, 4.0)

        if settings.DEBUG:
            individual.logger.info(
                'f_straight_movements {}'.format(pleasure))

        # pain - closer to an obstacle more pain
        pain = f_obstacle_dist(individual.sensor_activation)

        if settings.DEBUG:
            individual.logger.info('f_pain {}'.format(pain))

        #  fitness_t at time stamp
        fitness_t = V * pleasure * pain
        fitness_agg = np.append(fitness_agg, fitness_t)

        if settings.SAVE_DATA:
            with open(settings.PATH_NE + str(individual.genome_id) + '_fitness.txt', 'a') as f:
                f.write('{0!s},{1},{2},{3},{4},{5},{6},{7},{8}\n'.format(
                    individual.genome_id, scaled_output[0],
                    scaled_output[1], output[0], output[1],
                    V, pleasure, pain, fitness_t
                ))

    # behavarioral fitness function
    fitness_bff = [np.sum(fitness_agg)]

    # tailored fitness function
    fitness = fitness_bff[0]

    # Now send some data to V-REP in a non-blocking fashion:
    vrep.simxAddStatusbarMessage(
        client_id, 'fitness: {}'.format(fitness), vrep.simx_opmode_oneshot)

    # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    vrep.simxGetPingTime(client_id)

    print('%s genome_id: %d fitness: %f' %
          (t.getName(), individual.genome_id, fitness))

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
    vrep_abspath = '~/Developer/vrep-edu/vrep.app/Contents/MacOS/vrep'
    vrep_scene = os.getcwd() + '/arena.ttt'

    FNULL = open(os.devnull, 'w')
    # spawns multiple vrep instances
    vrep_servers = [Popen(
        ['{0} -gREMOTEAPISERVERSERVICE_{1}_TRUE_TRUE {2}'
            .format(vrep_abspath, port, vrep_scene)],
        shell=True, stdout=FNULL) for port in ports]

    time.sleep(10)

    clients = [vrep.simxStart(
        '127.0.0.1',
        port,
        True,
        True,
        5000,
        5) for port in ports]

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

    # Run for up to N generations.
    pe = ParrallelEvolution(clients, len(clients), eval_genome)
    winner = p.run(pe.evaluate, settings.N_GENERATIONS)

    # stop the workers
    pe.stop()

    print('\nBest genome:\n{!s}'.format(winner))

    # stop vrep simulation
    _ = [vrep.simxFinish(client) for client in clients]
    # kill vrep instances
    _ = [server.kill() for server in vrep_servers]

    node_names = {-1: 'A', -2: 'B', -3: 'C', -4: 'D', -5: 'E',
                  -6: 'F', -7: 'G', -8: 'H', -9: 'I', -10: 'J',
                  -11: 'K', -12: 'L', -13: 'M', -14: 'N', -15: 'O',
                  -16: 'P', 0: 'LEFT', 1: 'RIGHT', }

    visualize.draw_net(config, winner, True, node_names=node_names,
                       filename=settings.PATH_NE+'network')

    visualize.plot_stats(stats, ylog=False, view=False,
                         filename=settings.PATH_NE+'feedforward-fitness.svg')
    visualize.plot_species(
        stats, view=False, filename=settings.PATH_NE+'feedforward-speciation.svg')

    visualize.draw_net(config, winner, view=False, node_names=node_names,
                       filename=settings.PATH_NE+'winner-feedforward.gv')
    visualize.draw_net(config, winner, view=False, node_names=node_names,
                       filename=settings.PATH_NE+'winner-feedforward-enabled.gv', show_disabled=False)
    visualize.draw_net(config, winner, view=False, node_names=node_names,
                       filename=settings.PATH_NE+'winner-feedforward-enabled-pruned.gv', show_disabled=False, prune_unused=False)


if __name__ == '__main__':

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.ini')
    run(config_path)
