from common import *

from game import *

class NNModel:
    def __init__(self, device, seed, model):
        self.device = device
        self.model = model
        self.seed = seed

    def test(self, x):
        model_action = torch.zeros(FRAMES_PER_ACTION, 4).to(self.device)

        # l_x = xth layer in model
        self.l_2 = self.model[:4](x)
        self.l_3 = self.model[4:8](self.l_2)
        self.l_4 = self.model[8:12](self.l_3)
        self.l_5 = self.model[13:15](self.l_4.flatten())
        self.l_6 = self.model[15:17](self.l_5)
        self.l_7 = self.model[17:19](self.l_6)
        y = self.model[19:](self.l_7)
        self.pred = [float(f) for f in y]

        for i in range(FRAMES_PER_ACTION):
            m = -1
            m_elem = 0
            for j in range(4):
                if self.pred[i * 4 + j] > ACTION_THRESHOLD: model_action[i][j] = 1
            # prevents model from pressing opposite action (wouldn't move either direction)
            for j in range(0, len(model_action), 2):
                if model_action[i][j] == 1 and model_action[i][j + 1] == 1:
                    if self.pred[i * 4 + j] > self.pred[i * 4 + j + 1]:
                        model_action[i][j + 1] = 0
                    else:
                        model_action[i][j] = 0

        return model_action

    def validate(self):
        self.g = Game(self.device, self.seed)
        last_screen = self.g.get_screen()
        for f in range(round(TRAIN_FPS * TRAIN_TIME / FRAMES_PER_ACTION)):
            screen = self.g.get_screen()
            last_screen = self.g.Action_Update(self.test(torch.cat((last_screen, screen), 0)), get_screen=True, validate=TRAIN_MODEL)
        return self.g.score

    def train(self):
        if not TRAIN_MODEL: # test model to show to user
            init_window(WIDTH, HEIGHT, "DanmakuPOC-Q")
            set_target_fps(GUI_FPS)
        self.g = Game(self.device, self.seed)

        if not TRAIN_MODEL:
            last_screen = self.g.get_screen()
            for f in range(round(TRAIN_FPS * TRAIN_TIME / FRAMES_PER_ACTION)):
                screen = self.g.get_screen()
                last_screen = self.g.Action_Update(
                    self.test(torch.cat((last_screen, screen), 0)),
                    self.l_2,
                    self.l_3,
                    self.l_4,
                    self.l_5,
                    self.l_6,
                    self.l_7,
                    self.pred,
                    get_screen=True
                )

        return {
            'fitness': self.validate() if TRAIN_MODEL else self.g.score, # score by validation or model
            'hits': self.g.collide_count[2],
            'grazes': self.g.collide_count[1],
            'nears': self.g.collide_count[0]
        }

        return model_action
