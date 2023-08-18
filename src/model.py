from common import *

from game import *

import queue

class NNModel:
    device = -1
    index = -1
    g = -1
    model = -1

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
        for j in range(FPS * TRAIN_TIME):
            screen = self.g.get_screen()
            self.g.Update(self.test(screen))

            if self.index == TEST_MODEL:
                begin_drawing()
                self.g.Draw()
                end_drawing()

        return self.g.score
        
    def test(self, x):
        keys = [0 for i in range(4)]
        y = self.model(x)

        for i in range(4):
            if float(y[i]) > 0: keys[i] = 1

        return keys
