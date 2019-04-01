import vrep
from datetime import datetime
import os


def init():
    global RUNTIME
    global DEBUG
    global OP_MODE
    global PATH_NE
    global N_GENERATIONS
    global SAVE_DATA
    OP_MODE = vrep.simx_opmode_oneshot_wait
    PATH_NE = './data/neat/' + datetime.now().strftime('%Y-%m-%d') + '/'
    RUNTIME = 30
    DEBUG = False
    N_GENERATIONS = 2
    SAVE_DATA = False

    if not os.path.exists(PATH_NE):
        os.makedirs(PATH_NE)
