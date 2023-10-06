from common import *

from bullet import *
from dataset import *
from player import *

from pyray import *
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import time # REMOVE ZZZZZZZZ

class Game:
    def __init__(self, device, seed, player=False, bullets=False, score=0, frame_count=False, collide_count=False):
        self.device = device
        self.seed = seed
        self.rng = random.Random(seed)
        self.score = score
        self.colliding = [False, False, False] # 0 = near player, 1 = grazing player, 2 = touching player
        self.bullets =  [Bullet(*b) for b in bullets] if bullets else []
        if player:
            self.player = Player(*player)
        else:
            self.player = Player(WIDTH // 2, HEIGHT // 2 if BULLET_TYPE == BULLET_HONE else HEIGHT - 64 , PLAYER_SIZE)
        if not TRAIN_MODEL:
            self.untouched_count = 0 # -1 = touching bullets (any hitbox layer), 0 to (GAME_FPS * 2.5 - 1) = not touching, GAME_FPS * 2.5 = end and point reward
            self.collides = [] # shows collisions (used for demonstration, not in training)
            self.collide_count = collide_count if collide_count else [0 for i in range(3)] # no of frames each hitbox is touched

        if frame_count:
            self.frame_count = frame_count if BULLET_TYPE == BULLET_HONE else GAME_FPS // NUM_BULLETS - 1 # used to fire bullets at a constant rate
        else:
            self.frame_count = 0
            for i in range(NUM_BULLETS):
                self.bullets.append(self.new_bullet(BULLET_TYPE))

    def copy(self):
        g = Game(self.device, self.seed)
        g.player = self.player
        g.bullets = [self.bullets[i].copy() for i in range(len(self.bullets))]
        g.score = self.score
        g.frame_count = self.frame_count
        if not TRAIN_MODEL:
            g.collide_count = self.collide_count
        return g

    # Linux (and possibly macOS) can serialise objects to use w/ multiprocessing; Windows doesn't support this, so the needed members are exported and used to create an identical Game() instance
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
            False if TRAIN_MODEL else self.collide_count
        )

    def Reset(self, seed):
        self.rng = random.Random(seed)
        if BULLET_TYPE == BULLET_HONE: self.player = Player(WIDTH // 2, HEIGHT // 2, PLAYER_SIZE)
        else: self.player = Player(WIDTH // 2, HEIGHT - 64, PLAYER_SIZE)
        self.player.sprite = load_texture("reimu.png")

        self.bullets.clear()

        if BULLET_TYPE == BULLET_HONE:
            self.frame_count = GAME_FPS // NUM_BULLETS - 1 # used to fire bullets at a constant rate
        else:
            for i in range(NUM_BULLETS):
                self.bullets.append(self.new_bullet(BULLET_TYPE))

    def Sim_Update(self, action): # simulates an update and returns change in fitnesss
        # create copies of changed variables in Update() to rollback once Sim_Update() finishes
        _player = self.player.copy()
        _bullets = [self.bullets[i].copy() for i in range(len(self.bullets))]
        _score = self.score
        _frame_count = self.frame_count

        fitness = self.Action_Update(action)

        # restore copies of changed variables
        self.player = _player.copy()
        self.bullets = [_bullets[i].copy() for i in range(len(_bullets))]
        self.score = _score
        self.frame_count = _frame_count
        return fitness

    def Action_Update(self, action, l_2=0, l_3=0, l_4=0, l_5=0, l_6=0, l_7=0, pred=0, get_screen=False, validate=False): # action is FRAME_PER_ACTION frames of input; get_screen if True will also return the screen before the last frame (used for expected inputs in Model)
        i = -1
        for key in action:
            i += 1
            if get_screen and i == len(action) - 1:
                last_screen = self.get_screen() # used for past context
            # converts int representation into one-hot vector as input
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
                    validate=validate
                )
        return last_screen if get_screen else self.score
    
    def Update(self, keys, l_2=0, l_3=0, l_4=0, l_5=0, l_6=0, l_7=0, pred=0, validate=False): # extra paramters used when not TRAIN_MODEL
        if not TRAIN_MODEL:
            self.keys = keys
            self.collides = []
            if self.untouched_count < GAME_FPS * 2 + 1:
                self.untouched_count += 1

        self.colliding = [False for i in range(len(self.colliding))]
            
        self.player.Update(keys)
        for i in range(len(self.bullets) - 1, -1, -1): # iterates backwards so deletion of a bullet keeps matching indexes for next iterating bullets
            self.bullets[i].Update()
            if self.bullets[i].pos.x <= self.bullets[i].pos.width * -1 or \
                self.bullets[i].pos.x >= WIDTH or \
                self.bullets[i].pos.y <= self.bullets[i].pos.height * -1 or \
                self.bullets[i].pos.y >= HEIGHT:
                    if BULLET_TYPE == BULLET_RANDOM:
                        self.bullets[i] = self.new_bullet(BULLET_TYPE)
                    elif BULLET_TYPE == BULLET_HONE:
                        del self.bullets[i]
                        continue

        for i in range(len(self.bullets)):
            if self.is_colliding(Rectangle(
                self.player.pos.x - round(self.player.pos.width * 4),
                self.player.pos.y - round(self.player.pos.height * 4),
                self.player.pos.width * 9,
                self.player.pos.height * 9),
                self.bullets[i].pos):
                if self.colliding[0] == False:
                    self.colliding[0] = True
                    self.score -= 4
                if not TRAIN_MODEL:
                    self.collides.append(i);
                    self.collide_count[0] += 1
            if self.is_colliding(Rectangle(
                self.player.pos.x - round(self.player.pos.width * 1.5),
                self.player.pos.y - round(self.player.pos.height * 1.5),
                self.player.pos.width * 4,
                self.player.pos.height * 4),
                self.bullets[i].pos):
                if self.colliding[1] == False:
                    self.colliding[1] = True
                    self.score -= 16
                if not TRAIN_MODEL:
                    self.collides.append(i);
                    self.collide_count[1] += 1
            if self.is_colliding(self.player.pos, self.bullets[i].pos):
                if self.colliding[2] == False:
                    self.colliding[2] = True
                if not TRAIN_MODEL:
                    self.score -= 64
                    self.untouched_count = 0 # reset "untouched" count (bullet hits player)
                    self.collides.append(i);
                    self.collide_count[2] += 1
                else:
                    self.score -= 64 if validate else 999999 # high penalty prevents q-learning agent from even considering touching a bullet

        if BULLET_TYPE == BULLET_HONE:
            self.frame_count += 1
            if self.frame_count == GAME_FPS // NUM_BULLETS:
                self.frame_count = 0
                self.bullets.append(self.new_bullet(BULLET_HONE))

        if not TRAIN_MODEL:
            if self.untouched_count == GAME_FPS * 2 + 1:
                self.score += 1

            begin_drawing()
            self.Draw(
                l_2,
                l_3,
                l_4,
                l_5,
                l_6,
                l_7,
                pred
            )
            draw_fps(8, 8)
            end_drawing()

        return self.score

    def Draw(self, l_2, l_3, l_4, l_5, l_6, l_7, pred):
        clear_background(BLACK)
        
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
#        for i in range(l_2.shape[0]):
#            for y in range(l_2.shape[1]):
#                for x in range(l_2.shape[2]):
#                    c = round(max(min(float(l_2[i, y, x]), 1), 0) * 255)
#                    col = Color( c, c, c, 255 )
#                    draw_rectangle(264 + (i // 8) * 32 + x * 2, (i % 8) * 32 + y * 2, 2, 2, col)
#        for i in range(l_3.shape[0]):
#            for y in range(l_3.shape[1]):
#                for x in range(l_3.shape[2]):
#                    c = round(max(min(float(l_3[i, y, x]), 1), 0) * 255)
#                    col = Color( c, c, c, 255 )
#                    draw_rectangle(336 + (i // 16) * 16 + x * 2, (i % 16) * 16 + y * 2, 2, 2, col)
#        for i in range(l_4.shape[0]):
#            for y in range(l_4.shape[1]):
#                for x in range(l_4.shape[2]):
#                    c = round(max(min(float(l_4[i, y, x]), 1), 0) * 255)
#                    col = Color( c, c, c, 255 )
#                    draw_rectangle(408 + (i // 32) * 8 + x * 2, (i % 32) * 8 + y * 2, 2, 2, col)
#        for y in range(l_5.shape[0]):
#            col = Color( c, c, c, 255 )
#            c = round(max(min(float(l_5[y]), 1), 0) * 255)
#            draw_rectangle(480 + (y // 64) * 4, (y % 64) * 4, 4, 4, col)
#        for y in range(l_6.shape[0]):
#            col = Color( c, c, c, 255 )
#            c = round(max(min(float(l_6[y]), 1), 0) * 255)
#            draw_rectangle(552 + (y // 64) * 4, (y % 64) * 4, 4, 4, col)
#        for y in range(l_7.shape[0]):
#            col = Color( c, c, c, 255 )
#            c = round(max(min(float(l_7[y]), 1), 0) * 255)
#            draw_rectangle(576, y * 4, 4, 4, col)

        draw_text(str(self.score), 8, 32, 32, WHITE)

        for i in range(len(self.collide_count)):
            draw_text(str(self.collide_count[i]), 8, 96 + i * 32, 32, Color( 255, 255, 255, 128 ))

        k = {
            0: "U",
            1: "D",
            2: "L",
            3: "R",
        }
        for i in k:
            p = round(max(min(pred[i], 1), -1) * 96)
            if self.keys[i] == 1:
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

                    b_y = BULLET_SIZE * -1 + 1
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

                    b_x = BULLET_SIZE * -1 + 1
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

    def is_colliding(self, r1, r2):
        if (r1.x == r2.x or \
            (r1.x < r2.x and r1.x + r1.width > r2.x) or \
            (r1.x > r2.x and r2.x + r2.width > r1.x)) and \
            (r1.y == r2.y or \
            (r1.y < r2.y and r1.y + r1.height > r2.y) or \
            (r1.y > r2.y and r2.y + r2.height > r1.y)):
                return True
        return False

    def get_screen(self, bullet=False): # if bullet, also return whether or not a bullet is in screen (discard from training dataset if none)
        # creates 2D tensor (32x32) indicating location of bullets (0 = no bullet, 1 = bullet)
        # a half of the dimensions of the screen around the player is used (therefore dimensions of screen is used)
        # centre (top-left corner of (16, 16)) is player
        s = torch.zeros(1, 32, 32).to(self.device)
        p_x = self.player.pos.x + self.player.pos.width / 2
        p_y = self.player.pos.y + self.player.pos.height / 2
        if bullet:
            bullet_on_screen = False
        
        # first dimension
        for i in range(len(self.bullets)):
            # for bullets as well
            b_x = self.bullets[i].pos.x + self.bullets[i].pos.width / 2
            b_y = self.bullets[i].pos.y + self.bullets[i].pos.height / 2
            if b_x - p_x >= WIDTH / -2 and b_x - p_x < WIDTH / 2 and \
                b_y - p_y >= HEIGHT / -2 and b_y - p_y < HEIGHT / 2:
                x = math.floor((((b_x - p_x) / (WIDTH / 2)) + 1) * 16)
                y = math.floor((((b_y - p_y) / (HEIGHT / 2)) + 1) * 16)
                for y_2 in range(-2, 2):
                    for x_2 in range(-2, 2):
                        if abs(x_2 + 0.5) == 1.5 and abs(y_2 + 0.5) == 1.5:
                            continue
                        s[0][max(min(y + y_2, 31), 0)][max(min(x + x_2, 31), 0)] = 0.75 if abs(x_2 + 0.5) == 1.5 or abs(y_2 + 0.5) == 1.5 else 1
                if bullet:
                    bullet_on_screen = True

        return (bullet_on_screen, s) if bullet else s
