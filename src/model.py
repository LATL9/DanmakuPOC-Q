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

    def __init__(self, device, index, model):
        self.device = device
        self.index = index
        self.model = model

    def train(self):
        self.g = Game(self.device)
        for j in range(FPS * TRAIN_TIME):
            self.g.Update(self.test(self.g.get_screen()))
        return self.g.score
        
    def test(self, x):
        keys = [0 for i in range(4)]
        x = self.model(x)

        m = 0
        for i in range(4):
            if float(x[0][i]) > float(x[0][m]): m = i
        keys[m] = 1

        if self.index == -1:
            print({
                0: "Up",
                1: "Down",
                2: "Left",
                3: "Right",
            }[m])

        return keys
