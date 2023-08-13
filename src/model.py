from common import *

from game import *

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            self.g.Update(self.test(self.g.get_screen()))
            if self.index == TEST_MODEL:
                begin_drawing()
                self.g.Draw()
                end_drawing()

        return self.g.score
        
    def test(self, x):
        keys = [0 for i in range(4)]
        x = self.model(x)

        m = 0
        for i in range(4):
            if float(x[i]) > float(x[m]): m = i
        keys[m] = 1

        if self.index == -1:
            print({
                0: "Up",
                1: "Down",
                2: "Left",
                3: "Right",
            }[m])

        return keys
