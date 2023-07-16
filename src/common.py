import threading

WIDTH = 800
HEIGHT = 800
NUM_BULLETS = 32
NUM_PROCESSES = 12
NUM_MODELS = 24

class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)