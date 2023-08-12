import threading

TEST_MODEL = 14 # -1 = train models, dont test; 0- = test xth model by index
WIDTH = 800
HEIGHT = 800
FPS = 30
NUM_BULLETS = 32
NUM_PROCESSES = 12
NUM_MODELS = 48 # must be divisible by 4 (to divide into exact quarters)
NUM_MODELS_PER_PROCESS = round(NUM_MODELS / NUM_PROCESSES)
TRAIN_TIME = 20 # seconds
MUTATION_POWER = 0.2 # measure mutations change model

class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)
