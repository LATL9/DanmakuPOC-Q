from common import *

from game import *
import os

class NNModel:
    def __init__(self, device, seed, model):
        self.device = device
        self.rng = random.Random(seed)
        self.model = model
        self.seed = seed
        self.g = Game(self.device, self.seed)

    def reset(self, seed):
        self.g.Reset(seed)

    def train(self):
        if not TRAIN_MODEL: # test model to show to user
            init_window(WIDTH, HEIGHT, "DanmakuPOC-Q")
            set_target_fps(FPS)

        exp_inps = [] # expected tensor inputs (get_screen())
        exp_outs = [] # expected tensor outputs (actions)
        q_table = [] # stores q-values for actions at the actual state
        max_q_value = -9e99 # stores max q-value for actions from state'
        arrows = {
            0: '↑',
            1: '↓',
            2: '←',
            3: '→'
        }

        for f in range(round(FPS * TRAIN_TIME / FRAMES_PER_ACTION)):
            screen = self.g.get_screen()
            if TRAIN_MODEL:
                exp_inps.append(screen)
                exp_outs.append(torch.zeros(FRAMES_PER_ACTION, 4).to(self.device))
                q_table.append([])

                action = [
                    [0 for i in range(4)]
                    for j in range(FRAMES_PER_ACTION)
                ] # current action
                action_dec = [0 for i in range(FRAMES_PER_ACTION)] # single frame of action; each number = index for action (0 = up, 1 = down, 2 = left, 3 = right)

                while action_dec[FRAMES_PER_ACTION - 1] != 4:
                    action = [
                        [0 for i in range(4)]
                        for j in range(FRAMES_PER_ACTION)
                    ]
                    for i in range(len(action_dec)):
                        action[i][action_dec[i]] = 1
                    q_table[-1].append(self.g.Sim_Update(action))

                    action_new = [
                        [0 for i in range(4)]
                        for j in range(FRAMES_PER_ACTION)
                    ] # current action for state'
                    action_dec_new = [0 for i in range(FRAMES_PER_ACTION)] # action for state'
                    max_q_value = -9e99

                    while action_dec_new[FRAMES_PER_ACTION - 1] != 4:
                        action_new = [
                            [0 for i in range(4)]
                            for j in range(FRAMES_PER_ACTION)
                        ]
                        for i in range(len(action_dec_new)):
                            action_new[i][action_dec_new[i]] = 1
                        max_q_value = max(max_q_value, self.g.Sim_Update(action + action_new))

                        action_dec_new[0] += 1
                        for i in range(FRAMES_PER_ACTION - 1):
                            if action_dec_new[i] == 4:
                                action_dec_new[i] = 0
                                action_dec_new[i + 1] += 1
                            else:
                                break
                    q_table[-1][-1] += LEARNING_RATE * (q_table[-1][-1] + DISCOUNT_RATE * max_q_value) # reward and q-value are the same at this point, so they're ommited from equation as they cancel each other out

                    action_dec[0] += 1
                    for i in range(FRAMES_PER_ACTION - 1):
                        if action_dec[i] == 4:
                            action_dec[i] = 0
                            action_dec[i + 1] += 1
                        else:
                            break

                max_q_value = max(q_table[-1])
                action_dec = self.rng.choice([index for (index, item) in enumerate(q_table[-1]) if item == max_q_value]) # index for selected action

                j = action_dec
                print("{}/{}: ".format(f + 1, round(FPS * TRAIN_TIME / FRAMES_PER_ACTION)), end='')
                for i in range(FRAMES_PER_ACTION - 1, -1 ,-1):
                    exp_outs[-1][i][j // pow(4, i)] = 1
                    print(arrows[j // pow(4, i)], end='')
                    j -= pow(4, i) * (j // pow(4, i))

                self.g.Action_Update(exp_outs[-1])
                exp_outs[-1] = exp_outs[-1].flatten() # must be 1D to calculate loss
                print(", q-value {}".format(max_q_value), end='\r')
            else:
                self.g.Action_Update(
                    self.test(screen),
                    self.l_2,
                    self.l_3,
                    self.l_4,
                    self.l_5,
                    self.pred
                )

        return {
            'fitness': self.g.score,
            'exp_inps': exp_inps,
            'exp_outs': exp_outs
        }
        
    def test(self, x):
        model_action = torch.zeros(FRAMES_PER_ACTION, 4).to(self.device)

        # l_x = xth layer in model
        self.l_2 = self.model[:4](x)
        self.l_3 = self.model[4:8](self.l_2).flatten() # batch inputs aren't used (adds an extra dimension), so flatten() used instead
        self.l_4 = self.model[9:11](self.l_3)
        self.l_5 = self.model[11:13](self.l_4)
        y = self.model[13:](self.l_5)
        self.pred = [float(f) for f in y]

        for i in range(FRAMES_PER_ACTION):
            for j in range(4):
                if self.pred[i * 4 + j] > 0: model_action[i][j] = 1
            # prevents model from pressing opposite action (wouldn't move either direction)
            for j in range(0, len(model_action), 2):
                if model_action[i][j] == 1 and model_action[i][j + 1] == 1:
                    if self.pred[i * 4 + j] > self.pred[i * 4 + j + 1]: model_action[i][j + 1] = 0
                    else: model_action[i][j] = 0

        return model_action
