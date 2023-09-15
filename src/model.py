from common import *

from game import *
import os

class NNModel:
    device = -1
    g = -1
    model = -1
    pred = -1

    def __init__(self, device, seed, model):
        self.device = device
        self.model = model
        self.seed = seed
        self.g = Game(self.device, self.seed)

    def reset(self, seed):
        self.g.Reset(seed)

    def train(self):
        q_table = [[]] # stores q-values for actions at the actual state
        q_table_new = [
            [0 for i in range(4)]
            for j in range(FRAMES_PER_ACTION)
        ] # stores q-values for actions at current state'
        action = [
            [0 for i in range(4)]
            for j in range(FRAMES_PER_ACTION)
        ] # current action
        action_frame = [0 for i in range(FRAMES_PER_ACTION)] # single frame of action; each number = index for action (0 = up, 1 = down, 2 = left, 3 = right)

        while action_frame[FRAMES_PER_ACTION - 1] != 4:
            action = [
                [0 for i in range(4)]
                for j in range(FRAMES_PER_ACTION)
            ]

            for i in range(len(action_frame)):
                action[i][action_frame[i]] = 1
            q_table[len(q_table) - 1].append(self.g.Sim_Update(action))

            action_frame[0] += 1
            for i in range(FRAMES_PER_ACTION - 1):
                if action_frame[i] == 4:
                    action_frame[i] = 0
                    action_frame[i + 1] += 1
                else:
                    break

        exit()

        for j in range(FPS * TRAIN_TIME):
            screen = self.g.get_screen()
            self.g.Update(self.test(screen))

            if TEST_MODEL != -1:
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
        if TEST_MODEL != -1:
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
