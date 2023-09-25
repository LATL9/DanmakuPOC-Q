import torch.multiprocessing as mp

TRAIN_MODEL = False # true = train, false = test
WIDTH = 800
HEIGHT = 800
FPS = 10

BULLET_SIZE = 12
PLAYER_SIZE = 8

NUM_BULLETS = 8
NUM_PROCESSES = mp.cpu_count()
TRAIN_TIME = 10 # seconds

BULLET_RANDOM = 0
BULLET_HONE = 64 # NUM_BULLETS represents number of bullets fired per second when BULLET_HONE is used
BULLET_HONE_SPEED = HEIGHT / (FPS * 3) # will be on screen for at most 3 secs
BULLET_TYPE = BULLET_HONE # current type

LEARNING_RATE = 2e-4
DISCOUNT_RATE = 0.75
FRAMES_PER_ACTION = 4 # must be > 1, otherwise code in train() in Model will break
