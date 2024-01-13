import torch.multiprocessing as mp

BUILD_DL = True
# True = use Q-learning agent to build DataLoader for training
# False = don't and do TRAIN_MODEL instead
TRAIN_MODEL = True
# True = train model (must be True if BUILD_DL is True)
# False = test model

WIDTH = 800
HEIGHT = 800
TRAIN_FPS = 12 # internal FPS to use when training
GUI_FPS = 12 # FPS used with Raylib (should be multiple of FPS)
GAME_FPS = TRAIN_FPS if TRAIN_MODEL else GUI_FPS

BULLET_SIZE = 12
PLAYER_SIZE = 8
# odd multiple of player's width and height
# one multiple represents player and rest halved each side
GRAZE_SIZE = 9 if BUILD_DL else 7 # graze collision box
TOUCH_SIZE = 19 if BUILD_DL else 13 # touch collision box

NUM_BULLETS = 3
NUM_PROCESSES = mp.cpu_count()
TRAIN_TIME = 32 # seconds

BULLET_RANDOM = 0
BULLET_HONE = 1 # NUM_BULLETS represents number of bullets fired per second when BULLET_HONE is used
BULLET_RANDOM_HONE = 2 # 50/50 chance that a bullet is either BULLET_RANDOM or BULLET_HONE
BULLET_HONE_SPEED = HEIGHT / (GAME_FPS * 3) # will be on screen for at most 3 secs
BULLET_TYPE = BULLET_RANDOM_HONE # current type

ACTION_THRESHOLD = 0.25
DISCOUNT_RATE = 0.9
FRAMES_PER_ACTION = 3 # must be > 1, otherwise code in train() in Model will break
LEARNING_RATE = 1e-5
