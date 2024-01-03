from common import *

from bullet import *
from player import *
from q_dataset import *

from pyray import *
import copy
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Game:
    def __init__(self, device, seed, player=False, bullets=False, score=0, frame_count=False, collide_count=0):
        self.device = device
        self.seed = seed
        self.rng = random.Random(seed)
        self.score = score
        self.colliding = False
        self.collide_count = 0
        self.player = Player(*player) if player else Player(
            WIDTH // 2, HEIGHT // 2 if BULLET_TYPE == BULLET_HONE else HEIGHT - 64, PLAYER_SIZE
        )
        self.bullets = [Bullet(*b) for b in bullets] if bullets else [self.new_bullet(BULLET_TYPE) for i in range(NUM_BULLETS)]
        if not BUILD_DL:
            self.untouched_count = 0 # -1 = touching bullets (any hitbox layer), 0 to (GAME_FPS * 2.5 - 1) = not touching, GAME_FPS * 2.5 = end and point reward
        if not TRAIN_MODEL:
            self.FEATURES = [16, 64, 128, 1024, 256, 64] # number of features per hidden layer
            self.FEATURES_X = [] # number of features per row (for Draw())
            self.collides = [] # shows collisions (used for demonstration, not in training)
        # used to fire bullets at a constant rate
        self.frame_count = (
            frame_count if BULLET_TYPE == BULLET_HONE else GAME_FPS // NUM_BULLETS - 1
        ) if frame_count else 0

    def copy(self):
        g = Game(self.device, self.seed)
        g.player = self.player.copy()
        g.bullets = [self.bullets[i].copy() for i in range(len(self.bullets))]
        g.score = self.score
        g.frame_count = self.frame_count
        if not BUILD_DL:
            g.collide_count = self.collide_count
        return g

    # Linux (and possibly macOS) can serialise objects to use w/ multiprocessing
    # Windows doesn't support this, so the needed members are exported and used to create an identical Game() instance
    def export(self): # get arguments to reinit object
        return (
            self.device,
            self.seed,
            (
                self.player.pos.x,
                self.player.pos.y,
                self.player.pos.width # or height, as both (should be) same value
            ),
            [(
                self.bullets[i].pos.x,
                self.bullets[i].pos.y,
                self.bullets[i].pos.width,
                self.bullets[i].pos.height,
                self.bullets[i].v_x,
                self.bullets[i].v_y
            ) for i in range(len(self.bullets))],
            self.score,
            self.frame_count,
            False if BUILD_DL else self.collide_count
        )

    def Reset(self, seed):
        self.rng = random.Random(seed)
        if BULLET_TYPE == BULLET_HONE: self.player = Player(WIDTH // 2, HEIGHT // 2, PLAYER_SIZE)
        else: self.player = Player(WIDTH // 2, HEIGHT - 64, PLAYER_SIZE)

        self.bullets.clear()

        if BULLET_TYPE == BULLET_HONE:
            self.frame_count = GAME_FPS // NUM_BULLETS - 1 # used to fire bullets at a constant rate
        else:
            for i in range(NUM_BULLETS):
                self.bullets.append(self.new_bullet(BULLET_TYPE))
    
    # stop_bullet_collision = return low score (-inf) if bullet collides with player at any point
    def Sim_Update(self, action, stop_bullet_collision=False): # simulates an update and returns change in fitnesss
        # create copies of changed variables in Update() to rollback once Sim_Update() finishes
        _rng = copy.copy(self.rng)
        _player = self.player.copy()
        _bullets = [self.bullets[i].copy() for i in range(len(self.bullets))]
        _score = self.score
        _frame_count = self.frame_count

        fitness = self.Action_Update(action, stop_bullet_collision=stop_bullet_collision)

        # restore copies of changed variables
        self.rng = copy.copy(_rng)
        self.player = _player.copy()
        self.bullets = [_bullets[i].copy() for i in range(len(_bullets))]
        self.score = _score
        self.frame_count = _frame_count
        return fitness
    
    # action is FRAME_PER_ACTION frames of input
    # get_screen returns the screen before the last frame (used for expected inputs in Model)
    def Action_Update(self, action, l_2=0, l_3=0, l_4=0, l_5=0, l_6=0, l_7=0, pred=0, stop_bullet_collision=False, get_screen=False, validate=False):
        i = -1
        for key in action:
            i += 1
            if get_screen and i == len(action) - 1:
                last_screen = self.get_screen() # used for past context
            # parses int representation into one-hot vector as input
            if type(key) is int:
                key = [1 if i == key else 0 for i in range(4)]
            for j in range(GAME_FPS // TRAIN_FPS):
                self.Update(
                    key,
                    l_2,
                    l_3,
                    l_4,
                    l_5,
                    l_6,
                    l_7,
                    pred,
                    frame=i,
                    validate=validate,
                )
            if stop_bullet_collision and self.score <= float('-inf'): # if bullet collides with player
                return float('-inf') # low score would've been returned even if Action_Update() finished

        return last_screen if get_screen else self.score
    
    def Update(self, keys, l_2=0, l_3=0, l_4=0, l_5=0, l_6=0, l_7=0, pred=0, frame=0, validate=False): # extra paramters used when not TRAIN_MODEL
        if not BUILD_DL:
            if self.untouched_count < GAME_FPS * 4 + 1:
                self.untouched_count += 1
            if not TRAIN_MODEL:
                self.keys = keys
                self.collides = []

        self.colliding = False

        self.player.Update(keys)
        for i in range(len(self.bullets) - 1, -1, -1): # iterates backwards so deletion of a bullet keeps matching indexes for next iterating bullets
            self.bullets[i].Update()
            if self.bullets[i].pos.x <= self.bullets[i].pos.width * -1 or \
                self.bullets[i].pos.x >= WIDTH or \
                self.bullets[i].pos.y <= self.bullets[i].pos.height * -1 or \
                self.bullets[i].pos.y >= HEIGHT:
                    if BULLET_TYPE == BULLET_RANDOM:
                        self.bullets[i] = self.new_bullet(BULLET_TYPE)
                    elif BULLET_TYPE in (BULLET_HONE, BULLET_RANDOM_HONE):
                        del self.bullets[i]
                        continue

        penalty = 0
        for i in range(len(self.bullets)):
            d = math.sqrt(
                pow(
                    self.bullets[i].pos.x + self.bullets[i].pos.width // 2 -
                    (self.player.pos.x + self.player.pos.width // 2),
                    2
                ) + 
                pow(
                    self.bullets[i].pos.y + self.bullets[i].pos.height // 2 -
                    (self.player.pos.y + self.player.pos.height // 2),
                    2
                )
            )
            if d < math.sqrt(2 * pow(800/3, 2)):
                penalty += pow(d, 2) * 1.4e-5 - d * 7.4e-3 + 1

            if self.is_colliding(self.player.pos, self.bullets[i].pos):
                if self.colliding == False:
                    self.colliding = True
                    if BUILD_DL:
                        self.score -= float('inf') # infinitely-high penalty prevents q-learning agent from even considering touching a bullet
                if not BUILD_DL:
                    self.untouched_count = 0 # reset "untouched" count (bullet hits player)
                    self.collide_count += 1
                    if not TRAIN_MODEL:
                        self.collides.append(i)

        if BULLET_TYPE in (BULLET_HONE, BULLET_RANDOM_HONE):
            self.frame_count += 1
            if self.frame_count == GAME_FPS // NUM_BULLETS:
                self.frame_count = 0
                self.bullets.append(self.new_bullet(BULLET_TYPE))

        if not BUILD_DL:
            if self.untouched_count == GAME_FPS * 4 + 1:
                penalty /= 2
            if not TRAIN_MODEL:
                begin_drawing()
                self.Draw(
                    l_2,
                    l_3,
                    l_4,
                    l_5,
                    l_6,
                    l_7,
                    pred,
                    frame
                )
                draw_fps(8, 8)
                end_drawing()

        self.score -= penalty
        return self.score

    def Draw(self, l_2, l_3, l_4, l_5, l_6, l_7, pred, frame):
        clear_background(BLACK)
        draw_rectangle(0, 200, 113, 5, GREEN)
        
        self.player.Draw()
        for i in range(len(self.bullets)): self.bullets[i].Draw()
        for i in range(len(self.collides)): self.bullets[self.collides[i]].Draw(True)

        # minimap
        s = self.get_screen()
        draw_rectangle(0, 0, 256, 256, Color( 128, 128, 128, 128 ))
        draw_rectangle(16 * 8, 16 * 8, 8, 8, Color( 255, 255, 255, 128 ))
        for y in range(32):
            for x in range(32):
                if s[0, y, x] > 0: draw_rectangle(x * 8, y * 8, 8, 8, Color( s[0, y, x] * 255, 0, 0, 128 ))

        # layers
#        if not self.FEATURES_X:
#            for i in range(len(self.FEATURES)):
#                self.FEATURES_X.append(self.FEATURES[i])
#                for j in range(1, self.FEATURES[i]):
#                    if not 64 % j and (self.FEATURES[i] // j) * (64 // j) <= 256:
#                        self.FEATURES_X[-1] = j
#                        break
#
#        layers = [l_2, l_3, l_4, l_5, l_6, l_7]
#        for i in range(len(layers)):
#            if len(layers[i].shape) == 1:
#                node_size = 64 // self.FEATURES_X[i]
#                offset = 264 + (i * 64)
#                for f in range(layers[i].shape[0]):
#                    c = round(max(min(float(layers[i][f]), 1), 0) * 255)
#                    draw_rectangle(
#                        offset + (f % self.FEATURES_X[i]) * node_size,
#                        (f // self.FEATURES_X[i]) * node_size,
#                        node_size,
#                        node_size,
#                        Color( c, c, c, 255)
#                    )
#            else: # len(layers[i].shape) == 3
#                feature_size = (64 // self.FEATURES_X[i])
#                node_size = feature_size // layers[i].shape[2]
#                for f in range(layers[i].shape[0]):
#                    offset = 264 + (i * 64) + (f % self.FEATURES_X[i]) * feature_size
#                    for y in range(layers[i].shape[1]):
#                        for x in range(layers[i].shape[2]):
#                            c = round(max(min(float(layers[i][f, y, x]), 1), 0) * 255)
#                            draw_rectangle(
#                                offset + x * node_size,
#                                (f // self.FEATURES_X[i]) * feature_size + y * node_size,
#                                node_size,
#                                node_size,
#                                Color( c, c, c, 255)
#                            )

        draw_text(str(self.score), 8, 32, 32, WHITE)

        draw_text(str(self.collide_count), 8, 96, 32, Color( 255, 255, 255, 128 ))

        k = {
            0: "U",
            1: "D",
            2: "L",
            3: "R",
        }
        for i in k:
            p = round(max(min(float(pred[i]), 1), -1) * 96)
            if float(self.keys[i]) > ACTION_THRESHOLD:
                col_text = Color( 0, 255, 0, 192 )
                col_rect = Color( 0, 255, 0, 192 )
                draw_rectangle(24 + i * 32, 684 - p, 24, p, col_rect)
            else:
                col_text = Color( 255, 255, 255, 192 )
                if p > 0:
                    col_rect = Color( 255, 255, 255, 192 )
                    draw_rectangle(24 + i * 32, 684 - p, 24, p, col_rect)
                else:
                    col_rect = Color( 255, 0, 0, 192 )
                    draw_rectangle(24 + i * 32, 684, 24, p * -1, col_rect)
            draw_text(k[i], 24 + i * 32, 668, 32, col_text)

    def new_bullet(self, b):
        if b == BULLET_RANDOM:
            return Bullet(
                self.rng.randint(0, WIDTH - 1),
                BULLET_SIZE * -1 + 1,
                BULLET_SIZE,
                BULLET_SIZE,
                round((self.rng.randint(0, 1) - 0.5) * 2) * self.rng.randint(1, 240 // GAME_FPS),
                self.rng.randint(1, 480 // GAME_FPS)
            )
        elif b == BULLET_HONE:
            # edge that bullet is fired from:
            # 0 = top, 1 = bottom, 2 = left-side, 3 = right-side
            origin_edge = self.rng.randint(0, 3)
            # need centre player pos instead of top-left corner
            p_x = self.player.pos.x + self.player.pos.width / 2
            p_y = self.player.pos.y + self.player.pos.height / 2

            # maths means all bullets will move at same speed regardless of direction
            # opp from Pythagoras' theorem not specified as opp = player y - 0 = player y
            if origin_edge < 2: # 0, 1
                b_x = self.rng.randint(0, WIDTH - 1)
                adj = p_x - b_x

                if origin_edge == 0:
                    hyp = pow(pow(adj, 2) + pow(p_y, 2), 0.5)

                    b_y = 0
                    v_y = p_y * BULLET_HONE_SPEED / hyp
                else: # 1
                    hyp = pow(pow(adj, 2) + pow(p_y - HEIGHT, 2), 0.5)

                    b_y = HEIGHT - 1
                    v_y = (p_y - HEIGHT) * BULLET_HONE_SPEED / hyp
                v_x = adj * BULLET_HONE_SPEED / hyp
            else: # 2, 3
                b_y = self.rng.randint(0, HEIGHT - 1)
                adj = p_y - b_y

                if origin_edge == 2:
                    hyp = pow(pow(adj, 2) + pow(p_x, 2), 0.5)

                    b_x = 0
                    v_x = p_x * BULLET_HONE_SPEED / hyp
                else: # 3
                    hyp = pow(pow(adj, 2) + pow(p_x - WIDTH, 2), 0.5)

                    b_x = WIDTH - 1
                    v_x = (p_x - WIDTH) * BULLET_HONE_SPEED / hyp
                v_y = adj * BULLET_HONE_SPEED / hyp

            return Bullet(
                b_x - BULLET_SIZE // 2,
                b_y - BULLET_SIZE // 2,
                BULLET_SIZE,
                BULLET_SIZE,
                v_x,
                v_y
            )
        elif b == BULLET_RANDOM_HONE:
            return self.new_bullet(self.rng.randint(BULLET_RANDOM, BULLET_HONE))
        else:
            return Bullet(0, 0, 0, 0, 0, 0)

    def is_colliding(self, r_1, r_2):
        if (r_1.x == r_2.x or \
            (r_1.x < r_2.x and r_1.x + r_1.width > r_2.x) or \
            (r_1.x > r_2.x and r_2.x + r_2.width > r_1.x)) and \
            (r_1.y == r_2.y or \
            (r_1.y < r_2.y and r_1.y + r_1.height > r_2.y) or \
            (r_1.y > r_2.y and r_2.y + r_2.height > r_1.y)):
                return True
        return False

    def get_screen(self, bullet=False): # if bullet, also return whether or not a bullet is in screen (discard from training dataset if none)
        # creates 2D tensor (32x32) indicating location of bullets (0 = no bullet, 1 = bullet)
        # a thhird of the dimensions of the screen around the player is used (therefore two-thirds of screen (x and y) is used)
        # centre (top-left corner of (16, 16)) is player
        s = torch.zeros(1, 32, 32).to(self.device)
        p_x = self.player.pos.x + self.player.pos.width / 2
        p_y = self.player.pos.y + self.player.pos.height / 2
        if bullet:
            bullet_on_screen = False
        
        # first dimension
        for i in range(len(self.bullets)):
            b_x = self.bullets[i].pos.x + self.bullets[i].pos.width / 2
            b_y = self.bullets[i].pos.y + self.bullets[i].pos.height / 2
            if b_x - p_x >= WIDTH / -3 and \
                b_x - p_x < WIDTH / 3 and \
                b_y - p_y >= HEIGHT / -3 and \
                b_y - p_y < HEIGHT / 3:
                x = math.floor((((b_x - p_x) / (WIDTH / 3)) + 1) * 16)
                y = math.floor((((b_y - p_y) / (HEIGHT / 3)) + 1) * 16)
                for y_2 in range(-2, 3):
                    for x_2 in range(-2, 3):
                        if pow(pow(x_2, 2) + pow(y_2, 2), 0.5) <= 2.5:
                            s[0][max(min(y + y_2, 31), 0)][max(min(x + x_2, 31), 0)] = 1
                if bullet:
                    bullet_on_screen = True

        return (bullet_on_screen, s) if bullet else s
