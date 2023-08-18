from common import *

from game import *

import queue

class NNModel:
    device = -1
    index = -1
    g = -1
    model = -1
    pred = -1

    def __init__(self, device, seed, index, model):
        self.device = device
        self.index = index
        self.model = model
        self.seed = seed
        self.g = Game(self.device, self.seed)

        if self.index == TEST_MODEL: # test model to show to user
            init_window(WIDTH, HEIGHT, "DanmakuPRC")
            set_target_fps(FPS)

    def train(self):
        screen = self.g.get_screen()

        for j in range(FPS * TRAIN_TIME):
            self.g.Update(self.test(screen))

            if self.index == TEST_MODEL:
                begin_drawing()
                self.g.Draw(
                    self.model[:3](screen),
                    self.pred
                )
                end_drawing()

        return self.g.score
        
    def test(self, x):
        keys = [0 for i in range(4)]
        y = self.model(x)
        self.pred = [float(f) for f in y]

        for i in range(len(self.pred)):
            if self.pred[i] > 0: keys[i] = 1
        # prevents model from pressing opposite keys (wouldn't move either direction)
        for i in range(0, len(keys), 2):
            if keys[i] == 1 and keys[i + 1] == 1:
                if self.pred[i] > self.pred[i + 1]: keys[i + 1] = 0
                else: keys[i] = 0

        return keys
