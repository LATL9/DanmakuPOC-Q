import torch.multiprocessing as mp

TEST_MODEL = 12 # -1 = train models, dont test; 0- = test xth model by index
WIDTH = 800
HEIGHT = 800
FPS = 30

NUM_BULLETS = 1
NUM_PROCESSES = 8
NUM_MODELS = 40 # must be divisible by 4 (to divide into exact quarters)
NUM_MODELS_PER_PROCESS = round(NUM_MODELS / NUM_PROCESSES)
TRAIN_TIME = 20 # seconds
MUTATION_POWER = 0.1 # measure mutations change model

BULLET_RANDOM = 0
BULLET_HONE = 1 # NUM_BULLETS represents number of bullets fired per second when BULLET_HONE is used
BULLET_HONE_SPEED = HEIGHT / (FPS * 2) # will be on screen for at most 2 secs
BULLET_TYPE = BULLET_HONE # current type
