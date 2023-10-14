from common import *

from game import *

def calc_q_value(game_dict, q_table_dict, index, start, end): # index = start in decimal; end of range of actions is exclusive
    for f in range(round(TRAIN_FPS * TRAIN_TIME / FRAMES_PER_ACTION)):
        while f == len(game_dict): pass # wait until current frame is ready
        i = index
        action = start.copy() # current action; each number = index for action (0 = up, 1 = down, 2 = left, 3 = right)
        g = Game(*game_dict[f]) # get current frame of game from train()

        while action != end:
            # new instance of Game means Update() can be run once for action and Sim_Update() for action_new
            _g = g.copy()
            q_value = _g.Action_Update(action) - g.score
            fitness = _g.score
            action_new = [0 for j in range(FRAMES_PER_ACTION)] # current action for state'
            max_q_value = -9e99

            while action_new[FRAMES_PER_ACTION - 1] != 4:
                max_q_value = max(max_q_value, _g.Sim_Update(action_new) - fitness)

                action_new[0] += 1
                for j in range(FRAMES_PER_ACTION - 1):
                    if action_new[j] == 4:
                        action_new[j] = 0
                        action_new[j + 1] += 1
                    else:
                        break

            q_value += DISCOUNT_RATE * max_q_value # reward and q-value are the same at this point, so they're ommited from equation as they cancel each other out
            q_table_dict[i] = q_value # write q_value to static dictionary

            i += 1
            action[0] += 1
            for j in range(FRAMES_PER_ACTION - 1):
                if action[j] == 4:
                    action[j] = 0
                    action[j + 1] += 1
                else:
                    break

class NNModel:
    def __init__(self, device, seed, model=-1):
        self.device = device
        self.rng = random.Random(seed)
        self.model = model
        self.seed = seed
        self.arrows = { 
            0: '↑',
            1: '↓',
            2: '←',
            3: '→'
        }

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
        results = {}

        if BUILD_DL:
            exp_inps = [] # expected tensor inputs (get_screen())
            exp_outs = [] # expected tensor outputs (actions)
            jobs = []
            manager = mp.Manager()
            q_table = [] # stores q-values for actions at the actual state
            max_q_value = -9e99 # stores max q-value for actions from state'

            self.game_dict = manager.dict()
            self.q_table_dict = manager.dict()
            jobs = []
            for i in range(NUM_PROCESSES):
                index = i * pow(4, FRAMES_PER_ACTION) // NUM_PROCESSES
                jobs.append(mp.Process(target=calc_q_value, args=(
                    self.game_dict,
                    self.q_table_dict,
                    index,
                    self.to_base4(index),
                    self.to_base4((i + 1) * pow(4, FRAMES_PER_ACTION) // NUM_PROCESSES),
                )))
            for i in range(NUM_PROCESSES):
                jobs[i].start()
        elif not TRAIN_MODEL: # test model to show to user # add not back zz
            init_window(WIDTH, HEIGHT, "DanmakuPOC-Q")
            set_target_fps(GUI_FPS)
        self.g = Game(self.device, self.seed)

        if BUILD_DL or not TRAIN_MODEL:
            last_screen = self.g.get_screen()
            for f in range(round(TRAIN_FPS * TRAIN_TIME / FRAMES_PER_ACTION)):
                bullet_on_screen, screen = self.g.get_screen(True)
                if BUILD_DL:
                    exp_inps.append(torch.cat((last_screen, screen), 0))
                    exp_outs.append(torch.zeros(FRAMES_PER_ACTION, 4).to(self.device))

                    self.game_dict[f] = self.g.export() # processes will start once next instance of Game() is available
                    while len(self.q_table_dict) != pow(4, FRAMES_PER_ACTION):
                        pass # wait until all processes are "complete" (are waiting for next frame)
                    q_table.append(list({k: v for k, v in sorted(self.q_table_dict.items(), key=lambda item: item[0])}.values()))
                    for i in range(pow(4, FRAMES_PER_ACTION)):
                        del self.q_table_dict[i]

                    max_q_value = max(q_table[-1])
                    exp_outs_dec = self.to_base4(self.rng.choice([index for (index, item) in enumerate(q_table[-1]) if item == max_q_value]))
                    actual_outs = [ # actual action inputted to Game (no estimated confidence)
                        [1 if exp_outs_dec[j] == i else 0 for i in range(4)]
                        for j in range(FRAMES_PER_ACTION)
                    ]
                    last_screen = self.g.Action_Update(actual_outs, get_screen=True)

                    # don't deal with calculating exp_outs if:
                    # - no bullet on screen (not useful training data)
                    # - action causes player to touch bullet (bad training data)
                    if not bullet_on_screen or max_q_value < -9e90:
                        del exp_inps[-1]
                        del exp_outs[-1]
                        continue

                    # estimate confidence for other directions
                    for i in range(FRAMES_PER_ACTION):
                        for j in range(4): # for each direction
                            min_q_value = 9e99 # lowest q_value (excluding extremely high q_values (therefore bullet touched player))
                            sum_q_value = 0 # total q_value (also excluding high q_values)
                            for k in range(j, len(q_table[-1]), pow(4, i + 1)):
                                for l in range(k, k + pow(4, i)):
                                    if q_table[-1][k] > -9e90:
                                        min_q_value = min(min_q_value, q_table[-1][k])
                                        sum_q_value += q_table[-1][k]
                            if min_q_value != 9e99: # if equal, all ways of moving in direction at frame cause bullet to touch player; confidence should be 0
                                if min_q_value == 0: # division by zero
                                    exp_outs[-1][i][j] = 0.75 if j == exp_outs_dec[i] else ACTION_THRESHOLD
                                else:
                                    exp_outs[-1][i][j] = -1 * ((sum_q_value / pow(4, FRAMES_PER_ACTION - 1)) - min_q_value) / min_q_value
                                    if j == exp_outs_dec[i]:
                                        exp_outs[-1][i][j] = max(0.75, exp_outs[-1][i][j] * ACTION_THRESHOLD)
                                    else:
                                        exp_outs[-1][i][j] *= ACTION_THRESHOLD
                                # confidence between min_q_value and 0 is mapped to between 0 and ACTION_THRESHOLD or 0-0.75 if choosed action

                    print("Frame {}/{}: ".format(f, round(TRAIN_FPS * TRAIN_TIME / FRAMES_PER_ACTION)), end='')
                    for i in range(FRAMES_PER_ACTION):
                        print(self.arrows[exp_outs_dec[i]], end='')
                    print(", Q-value {}{}".format(max_q_value, ' ' * 10), end='\r')

                    exp_outs[-1] = exp_outs[-1].flatten() # must be 1D to calculate loss
                elif not TRAIN_MODEL:
                    last_screen = self.g.Action_Update(
                        self.test(torch.cat((last_screen, screen), 0), bullet_on_screen),
                        self.l_2,
                        self.l_3,
                        self.l_4,
                        self.l_5,
                        self.l_6,
                        self.l_7,
                        self.pred,
                        get_screen=True
                    )
            q_fitness = self.g.score

        if BUILD_DL:
            results['q_fitness'] = q_fitness # score by Q-learning agent
            results['exp_inps'] = exp_inps
            results['exp_outs'] = exp_outs
        else: # if training (to validate) or testing
            results['fitness'] = self.validate() if TRAIN_MODEL else self.g.score # score by validation or model
        return results

    def validate(self): # should be run if TRAIN_MODEL (+ not BUILD_DL)
        self.g = Game(self.device, self.seed)
        last_screen = self.g.get_screen()
        for f in range(round(TRAIN_FPS * TRAIN_TIME / FRAMES_PER_ACTION)):
            screen = self.g.get_screen()
            last_screen = self.g.Action_Update(self.test(torch.cat((last_screen, screen), 0)), get_screen=True, validate=TRAIN_MODEL)
        return self.g.score

    def test(self, x, batches=False):
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
                    if self.pred[i * 4 + j] > self.pred[i * 4 + j + 1]: model_action[i][j + 1] = 0
                    else: model_action[i][j] = 0

        return model_action
