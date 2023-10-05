import torch.multiprocessing as mp

BUILD_DL = True
# True = use Q-learning agent to build DataLoader for training
# False = don't and do TRAIN_MODEL instead
TRAIN_MODEL = True 
# True = train model
# False = test model
WIDTH = 800
HEIGHT = 800
TRAIN_FPS = 12 # internal FPS to use when training
GUI_FPS = 36 # FPS used with Raylib (should be multiple of FPS)
GAME_FPS = TRAIN_FPS if BUILD_DL or TRAIN_MODEL else GUI_FPS

BULLET_SIZE = 12
PLAYER_SIZE = 8

NUM_BULLETS = 2
NUM_PROCESSES = mp.cpu_count()
TRAIN_TIME = 32 # seconds

BULLET_RANDOM = 0
BULLET_HONE = 1 # NUM_BULLETS represents number of bullets fired per second when BULLET_HONE is used
BULLET_HONE_SPEED = HEIGHT / (GAME_FPS * 1) # will be on screen for at most 3 secs
BULLET_TYPE = BULLET_HONE # current type

LEARNING_RATE = 1e-5
DISCOUNT_RATE = 0.9
FRAMES_PER_ACTION = 3 # must be > 1, otherwise code in train() in Model will break
