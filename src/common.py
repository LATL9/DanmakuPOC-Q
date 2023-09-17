TRAIN_MODEL = False # true = train, false = test
WIDTH = 800
HEIGHT = 800
FPS = 30

BULLET_SIZE = 12
PLAYER_SIZE = 8

NUM_BULLETS = 2
NUM_PROCESSES = 12
NUM_MODELS = 48 # must be divisible by 4 (to divide into exact quarters)
NUM_MODELS_PER_PROCESS = round(NUM_MODELS / NUM_PROCESSES)
TRAIN_TIME = 16 # seconds
MUTATION_POWER = 0.1 # measure mutations change model

BULLET_RANDOM = 0
BULLET_HONE = 1 # NUM_BULLETS represents number of bullets fired per second when BULLET_HONE is used
BULLET_HONE_SPEED = HEIGHT / (FPS * 3) # will be on screen for at most 3 secs
BULLET_TYPE = BULLET_HONE # current type

LEARNING_RATE = 1e-3
DISCOUNT_RATE = 0.75
FRAMES_PER_ACTION = 3 # must be > 1, otherwise code in train() in Model will break
