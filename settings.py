import vrep
from datetime import datetime

def init():
    global CLIENT_ID
    global RUNTIME
    global DEBUG
    global OP_MODE
    global PORT_NUM
    global PATH_EA
    global PATH_NE
    global MIN
    global MAX
    global POPULATION
    global N_GENERATIONS
    global CXPB
    global MUTPB
    global SAVE_DATA
    OP_MODE = vrep.simx_opmode_oneshot_wait
    PORT_NUM = 19997
    PATH_EA = './data/ea/' + datetime.now().strftime('%Y-%m-%d') + '/'
    PATH_NE = './data/neat/' + datetime.now().strftime('%Y-%m-%d') + '/'
    MIN = 0.0
    MAX = 3.0
    RUNTIME = 30
    CXPB = 0.1
    MUTPB = 0.2
    DEBUG = False
    N_GENERATIONS = 2
    POPULATION = 2
    SAVE_DATA = False
