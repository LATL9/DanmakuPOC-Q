from common import *

from game import *
import os

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

    def reset(self, seed):
        self.g.Reset(seed)

    def train(self):
        for j in range(FPS * TRAIN_TIME):
            screen = self.g.get_screen()
            self.test(screen)
            self.g.Update(self.test(screen))

            if self.index == TEST_MODEL:
                begin_drawing()
                self.g.Draw(
                    self.l_2,
                    self.l_3,
                    self.l_4,
                    self.l_5,
                    self.pred
                )
                draw_text(str(j), 8, 64, 32, WHITE)
                draw_fps(8, 8)
                end_drawing()

        return self.g.score
        
    def test(self, x):
        keys = [0 for i in range(4)]
        if self.index == TEST_MODEL:
            # l_x = xth layer in model
            self.l_2 = self.model[:4](x)
            self.l_3 = self.model[4:8](self.l_2)
            self.l_4 = self.model[8:11](self.l_3)
            self.l_5 = self.model[11:13](self.l_4)
            y = self.model[13:](self.l_5)
        else:
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
