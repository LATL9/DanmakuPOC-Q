from common import *

from game import *
from nn_model import *

def calc_q_value(game_dict, q_table_dict, index, start, end): # index = start in decimal; end of range of actions is exclusive
    for f in range(round(TRAIN_FPS * TRAIN_TIME / FRAMES_PER_ACTION)):
        while f == len(game_dict):
            pass # wait until current frame is ready
        # each number = index for action (0 = up, 1 = down, 2 = left, 3 = right, 4 = do nothing)
        ideal_action = False # whether or not an action with a Q-value > -inf (player doesn't hit bullet) has been found
        i = index
        action = start.copy() # current action
        g = Game(*game_dict[f]) # get current frame of game from train()
        q_values = {}

        while action != end:
            # new instance of Game means Update() can be run once for action and Sim_Update() for action_new
            _g = g.copy()
            q_value = _g.Action_Update(action, stop_bullet_collision=ideal_action) - g.score
            fitness = _g.score
            if ideal_action and q_value == float('-inf'):
                q_values[i] = float('-inf') # not actual Q-value, but would be ignored for low value anyways
            else:
                action_new = [0 for j in range(FRAMES_PER_ACTION)] # current action for state'
                max_q_value = float('-inf')

                while action_new[-1] != 5:
                    max_q_value = max(max_q_value, _g.Sim_Update(action_new, stop_bullet_collision=True) - fitness)

                    action_new[0] += 1
                    for j in range(FRAMES_PER_ACTION - 1):
                        if action_new[j] == 5:
                            action_new[j] = 0
                            action_new[j + 1] += 1
                        else:
                            break

                q_value += DISCOUNT_RATE * max_q_value # reward and q-value are the same at this point, so they're ommited from equation as they cancel each other out
                q_values[i] = q_value
                if not ideal_action and q_value > float('-inf'):
                    ideal_action = True # an "ideal" action (see instantiation of ideal_action) has been found

            i += 1
            action[0] += 1
            for j in range(FRAMES_PER_ACTION - 1):
                if action[j] == 5:
                    action[j] = 0
                    action[j + 1] += 1
                else:
                    break
        q_table_dict[index] = q_values.copy() # write Q-values to static dictionary

class QAgent:
    def __init__(self, device, seed):
        self.device = device
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.arrows = { 
            0: '↑',
            1: '↓',
            2: '←',
            3: '→',
            4: '-'
        }
        # rotates counterclockwise in line with PyTorch's rot90()
        # value -> key : key -> counterclockwise rotation
        self.rotations = [
            { # 90 degrees
                0: 3,
                1: 2,
                2: 0,
                3: 1
            }, { # 180 degrees
                0: 1,
                1: 0,
                2: 3,
                3: 2
            }, { # 270 degrees
                0: 2,
                1: 3,
                2: 1,
                3: 0
            }
        ]

    def to_base5(self, num):
        l = [0 for i in range(FRAMES_PER_ACTION)]
        for i in range(FRAMES_PER_ACTION - 1, -1 ,-1):
            l[i] = num // pow(5, i)
            num -= pow(5, i) * (num // pow(5, i))
            if num == 0:
                return l

    def train(self):
        results = {}

        exp_inps = [] # expected tensor inputs (get_screen())
        exp_outs = [] # expected tensor outputs (actions)
        jobs = []
        manager = mp.Manager()
        q_table = [] # stores q-values for actions at the actual state
        max_q_value = float('-inf') # stores max q-value for actions from state'

        self.game_dict = manager.dict()
        self.q_table_dict = manager.dict()
        jobs = []
        for i in range(NUM_PROCESSES):
            index = i * pow(5, FRAMES_PER_ACTION) // NUM_PROCESSES
            jobs.append(mp.Process(target=calc_q_value, args=(
                self.game_dict,
                self.q_table_dict,
                index,
                self.to_base5(index),
                self.to_base5((i + 1) * pow(5, FRAMES_PER_ACTION) // NUM_PROCESSES),
            )))
        for i in range(NUM_PROCESSES):
            jobs[i].start()

        self.g = Game(self.device, self.seed)

        last_screen = self.g.get_screen()
        for f in range(round(TRAIN_FPS * TRAIN_TIME / FRAMES_PER_ACTION)):
            bullet_on_screen, screen = self.g.get_screen(True)
            exp_inps.append(torch.cat((last_screen, screen), 0))
            exp_outs.append(torch.zeros(FRAMES_PER_ACTION, 4).to(self.device))
            
            self.game_dict[f] = self.g.export() # processes will start once next instance of Game() is available
            while len(self.q_table_dict) != NUM_PROCESSES:
                pass # wait until all processes are "complete" (are waiting for next frame)

            temp_dict = {}
            for d in self.q_table_dict:
                for i in self.q_table_dict[d]:
                    temp_dict[i] = self.q_table_dict[d][i]

            q_table.append(list({k: v for k, v in sorted(temp_dict.items(), key=lambda item: item[0])}.values()))
            self.q_table_dict.clear()

            max_q_value = max(q_table[-1])
            exp_outs_dec = self.to_base5(self.rng.choice([index for (index, item) in enumerate(q_table[-1]) if item == max_q_value]))
            actual_outs = [] # actual actions inputted into Game (no estimated confidence)
            for i in range(FRAMES_PER_ACTION):
                if exp_outs_dec[i] != 4: # else, do nothing
                    actual_outs.append([1 if exp_outs_dec[i] == j else 0 for j in range(4)])
                else:
                    actual_outs.append([0 for j in range(4)])

            last_screen = self.g.Action_Update(actual_outs, get_screen=True)

            # don't deal with calculating exp_outs if:
            # - no bullet on screen (unuseful training data)
            # - action causes player to touch bullet (bad training data)
            if not bullet_on_screen or max_q_value < float('-inf'):
                del exp_inps[-1]
                del exp_outs[-1]
                continue

            # estimate confidence for other directions
            for i in range(FRAMES_PER_ACTION):
                for j in range(4): # for each direction ("do nothing" direction not included)
                    min_q_value = float('inf') # infinitely-high q_value (if unchanged, bullet touched player)
                    sum_q_value = 0 # total q_value (also excluding high q_values)
                    for k in range(j, len(q_table[-1]), pow(5, i + 1)):
                        for l in range(k, k + pow(5, i)):
                            if q_table[-1][l] > float('-inf'):
                                min_q_value = min(min_q_value, q_table[-1][l])
                                sum_q_value += q_table[-1][l]
                    if min_q_value != float('inf'): # if equal, all ways of moving in direction at frame cause bullet to touch player; confidence should be 0
                        if min_q_value == 0: # division by zero
                            exp_outs[-1][i][j] = 0.75 if j == exp_outs_dec[i] else ACTION_THRESHOLD
                        else:
                            exp_outs[-1][i][j] = -ACTION_THRESHOLD * ((sum_q_value / pow(5, FRAMES_PER_ACTION - 1)) - min_q_value) / min_q_value
                            if j == exp_outs_dec[i]:
                                exp_outs[-1][i][j] = max(0.75, exp_outs[-1][i][j])
                        # confidence between min_q_value and 0 is mapped from 0 - ACTION_THRESHOLD or 0.75 - 1 if chosen action

            print("Frame {}/{}: ".format(f, round(TRAIN_FPS * TRAIN_TIME / FRAMES_PER_ACTION)), end='')
            for i in range(FRAMES_PER_ACTION):
                print(self.arrows[exp_outs_dec[i]], end='')
            print(", Q-value {}{}".format(max_q_value, ' ' * 10), end='\r')

            exp_outs[-1] = exp_outs[-1].flatten() # must be 1D to calculate loss

        q_fitness = self.g.score
        # extrapolate training data by rotating 3x (4x the data)
        for exp_inp, exp_out in zip(exp_inps[:len(exp_inps)], exp_outs[:len(exp_outs)]): # zip() faster than range() in this case
            for i in range(3):
                exp_inps.append(torch.rot90(exp_inp, i + 1, dims=[1, 2]))
                t = []
                for j in range(0, FRAMES_PER_ACTION * 4, 4):
                    for k in range(4):
                        t.append(exp_out[j + self.rotations[i][k]])
                exp_outs.append(torch.Tensor(t))

        return {
            'q_fitness': q_fitness, # score by Q-learning agent
            'exp_inps': exp_inps,
            'exp_outs': exp_outs
        }
