from common import *

from game import *

import os

def calc_q_value(q_table_dict, g, index, start, end): # index = start in decimal; end of range of actions is exclusive
    action = start # current action; each number = index for action (0 = up, 1 = down, 2 = left, 3 = right)
    g = Game(*g)

    while action != end:
        # new instance of Game means Update() can be run once for action and Sim_Update() for action_new
        _g = g.copy()
        fitness = _g.score
        q_value = _g.Action_Update(action) - fitness
        action_new = [0 for i in range(FRAMES_PER_ACTION)] # current action for state'
        max_q_value = -9e99

        while action_new[FRAMES_PER_ACTION - 1] != 4:
            max_q_value = max(max_q_value, _g.Sim_Update(action_new) - fitness)

            action_new[0] += 1
            for i in range(FRAMES_PER_ACTION - 1):
                if action_new[i] == 4:
                    action_new[i] = 0
                    action_new[i + 1] += 1
                else:
                    break

        q_value += LEARNING_RATE * (q_value + DISCOUNT_RATE * max_q_value) # reward and q-value are the same at this point, so they're ommited from equation as they cancel each other out
        q_table_dict[index] = q_value # write q_value to static dictionary

        index += 1
        action[0] += 1
        for i in range(FRAMES_PER_ACTION - 1):
            if action[i] == 4:
                action[i] = 0
                action[i + 1] += 1
            else:
                break

class NNModel:
    def __init__(self, device, seed, model):
        self.device = device
        self.rng = random.Random(seed)
        self.model = model
        self.seed = seed
        self.g = Game(self.device, self.seed)

    def reset(self, seed):
        self.g.Reset(seed)

    def to_base4(self, num):
        l = [0 for i in range(FRAMES_PER_ACTION)]
        for i in range(FRAMES_PER_ACTION - 1, -1 ,-1):
            l[i] = num // pow(4, i)
            num -= pow(4, i) * (num // pow(4, i))
            if num == 0:
                return l


    def train(self):
        if TRAIN_MODEL:
            jobs = []
            q_table_manager = mp.Manager()
            arrows = {
                0: '↑',
                1: '↓',
                2: '←',
                3: '→'
            }

            exp_inps = [] # expected tensor inputs (get_screen())
            exp_outs = [] # expected tensor outputs (actions)
            q_table = [] # stores q-values for actions at the actual state
            max_q_value = -9e99 # stores max q-value for actions from state'
        else: # test model to show to user
            init_window(WIDTH, HEIGHT, "DanmakuPOC-Q")
            set_target_fps(FPS)

        for f in range(round(FPS * TRAIN_TIME / FRAMES_PER_ACTION)):
            screen = self.g.get_screen()
            g_export = self.g.export()
            if TRAIN_MODEL:
                exp_inps.append(screen)
                exp_outs.append(torch.zeros(FRAMES_PER_ACTION, 4).to(self.device))

                self.q_table_dict = q_table_manager.dict()
                jobs = list()
                for i in range(NUM_PROCESSES):
                    index = i * pow(4, FRAMES_PER_ACTION) // NUM_PROCESSES
                    jobs.append(mp.Process(target=calc_q_value, args=(
                        self.q_table_dict,
                        g_export,
                        index,
                        self.to_base4(index),
                        self.to_base4((i + 1) * pow(4, FRAMES_PER_ACTION) // NUM_PROCESSES),
                    )))
                for i in range(NUM_PROCESSES):
                    jobs[i].start()
                for i in range(NUM_PROCESSES):
                    jobs[i].join()

                q_table.append(list({k: v for k, v in sorted(self.q_table_dict.items(), key=lambda item: item[0])}.values()))

                print("Frame {}/{}: ".format(f, round(FPS * TRAIN_TIME / FRAMES_PER_ACTION)), end='')
                max_q_value = max(q_table[-1])
                exp_outs_dec = self.to_base4(self.rng.choice([index for (index, item) in enumerate(q_table[-1]) if item == max_q_value]))
                for i in range(FRAMES_PER_ACTION):
                    exp_outs[-1][i][exp_outs_dec[i]] = 1
                    print(arrows[exp_outs_dec[i]], end='')
                print(", Q-value {}".format(max_q_value), end='\r')

                self.g.Action_Update(exp_outs[-1])
                exp_outs[-1] = exp_outs[-1].flatten() # must be 1D to calculate loss
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
