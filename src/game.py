from common import *

from bullet import *
from player import *

from pyray import *
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Game:
    device = -1
    rng = -1
    bullets = []
    player = Player(-1, -1, -1)
    score = 0
    colliding = [False, False, False] # 0 = near player, 1 = grazing player, 2 = touching player
    still_colliding = [False, False, False]
    invincible_count = [-1, -1, -1] # -1 = not invincible, 0 to (FPS - 1) = invincible frame (FPS is end)
    untouched_count = -1 # -1 = touching bullets (any hitbox layer), 0 to (FPS * 2 - 1) = not touching, FPS * 2= end and point reward
    if TEST_MODEL != -1: collides = [] # shows collisions (used for demonstration, not in training)

    def __init__(self, device, seed):
        self.device = device
        self.rng = random.Random(seed)
        if BULLET_TYPE == BULLET_HONE: self.player = Player(WIDTH // 2, HEIGHT // 2, PLAYER_SIZE)
        else: self.player = Player(WIDTH // 2, HEIGHT - 64, PLAYER_SIZE)
        self.score = 0

        # edge barrier
        self.bullets = []
        self.bullets.append(Bullet(
            0,
            0,
            BULLET_SIZE,
            0,
            0
        ))
        self.bullets[0].pos.width = WIDTH
        self.bullets.append(Bullet(
            0,
            HEIGHT - BULLET_SIZE,
            BULLET_SIZE,
            0,
            0
        ))
        self.bullets[1].pos.width = WIDTH
        self.bullets.append(Bullet(
            0,
            BULLET_SIZE,
            BULLET_SIZE,
            0,
            0
        ))
        self.bullets[2].pos.height = HEIGHT - BULLET_SIZE * 2
        self.bullets.append(Bullet(
            WIDTH - BULLET_SIZE,
            BULLET_SIZE,
            BULLET_SIZE,
            0,
            0
        ))
        self.bullets[3].pos.height = HEIGHT - BULLET_SIZE * 2

        if BULLET_TYPE == BULLET_HONE:
            self.frame_count = FPS // NUM_BULLETS - 1 # used to fire bullets at a constant rate
        else:
            for i in range(NUM_BULLETS):
                self.bullets.append(self.new_bullet(BULLET_TYPE))

    def End(self): return self.score
    
    def Update(self, keys):
        #keys = [is_key_down(KEY_UP), is_key_down(KEY_DOWN), is_key_down(KEY_LEFT), is_key_down(KEY_RIGHT)]
        if TEST_MODEL != -1:
            self.keys = keys
            self.collides = []

        self.colliding = [False for i in range(len(self.colliding))]

        invincible = False
        for i in range(len(self.invincible_count)):
            if self.invincible_count[i] != -1:
                self.invincible_count[i] += 1
                if self.invincible_count[i] == FPS + 1: self.invincible_count[i] = -1
                invincible = True
        if invincible: self.untouched_count = 0
        else:
            if self.untouched_count < FPS * 2 + 1: self.untouched_count += 1
            
        self.player.Update(keys)
        for i in range(len(self.bullets) - 1, -1, -1): # iterates backwards so deletion of a bullet keeps matching indexes for next iterating bullets
            self.bullets[i].Update()
            if self.bullets[i].pos.x <= self.bullets[i].pos.width * -1 or \
                self.bullets[i].pos.x >= WIDTH or \
                self.bullets[i].pos.y <= self.bullets[i].pos.height * -1 or \
                self.bullets[i].pos.y >= HEIGHT:
                    if BULLET_TYPE == BULLET_RANDOM: self.bullets[i] = self.new_bullet(BULLET_TYPE)
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
                self.colliding[0] = True
                if self.invincible_count[0] == -1: self.invincible_count[0] = 0
                self.untouched_count = 0
                if TEST_MODEL != -1: self.collides.append(i);
            if self.is_colliding(Rectangle(
                self.player.pos.x - round(self.player.pos.width * 2),
                self.player.pos.y - round(self.player.pos.height * 2),
                self.player.pos.width * 5,
                self.player.pos.height * 5),
                self.bullets[i].pos):
                self.colliding[1] = True
                if self.invincible_count[1] == -1: self.invincible_count[1] = 0
                self.untouched_count = 0
                if TEST_MODEL != -1: self.collides.append(i);
            if self.is_colliding(self.player.pos, self.bullets[i].pos):
                self.colliding[2] = True
                if self.invincible_count[2] == -1: self.invincible_count[2] = 0
                self.untouched_count = 0
                if TEST_MODEL != -1: self.collides.append(i);
        
        for i in range(len(self.invincible_count)):
            if self.invincible_count[i] == 0: self.score -= (i + 1) * FPS
        if self.untouched_count == FPS * 2 + 1: self.score += 1

        if BULLET_TYPE == BULLET_HONE:
            self.frame_count += 1
            if self.frame_count == FPS // NUM_BULLETS:
                self.frame_count = 0
                self.bullets.append(self.new_bullet(BULLET_HONE))

    def Draw(self, pred):
        clear_background(BLACK)
        
        self.player.Draw()
        for i in range(len(self.bullets)): self.bullets[i].Draw()
        for i in range(len(self.collides)): self.bullets[self.collides[i]].Draw(True)

        # minimap
        s = self.get_screen()
        draw_rectangle(0, 0, 256, 256, Color( 128, 128, 128, 192 ))
        draw_rectangle(16 * 8, 16 * 8, 8, 8, Color( 255, 255, 255, 192 ))
        for y in range(32):
            for x in range(32):
                if s[0, y, x] == 1:  draw_rectangle(x * 8, y * 8, 8, 8, Color( 255, 0, 0, 192 ))

        draw_text(str(self.score), 8, 32, 32, WHITE)
        if self.invincible_count != -1: draw_text(str(self.invincible_count), 8, 64, 32, DARKGRAY)

        k = {
            0: "U",
            1: "D",
            2: "L",
            3: "R",
        }
        for i in k:
            if self.keys[i] == 1:
                col_text = Color( 0, 255, 0, 192 )
                col_rect = Color( 0, 255, 0, 192 )
                draw_rectangle(24 + i * 32, 684 - round(pred[i] * 96), 24, round(pred[i] * 96), col_rect)
            else:
                col_text = Color( 255, 255, 255, 192 )
                if pred[i] > 0:
                    col_rect = Color( 255, 255, 255, 192 )
                    draw_rectangle(24 + i * 32, 684 - round(pred[i] * 96), 24, round(pred[i] * 96), col_rect)
                else:
                    col_rect = Color( 255, 0, 0, 192 )
                    draw_rectangle(24 + i * 32, 684, 24, round(pred[i] * -96), col_rect)
            draw_text(k[i], 24 + i * 32, 668, 32, col_text)

    def new_bullet(self, b):
        if b == BULLET_RANDOM:
            return Bullet(
                self.rng.randint(0, WIDTH - 1),
                BULLET_SIZE * -1 + 1,
                BULLET_SIZE,
                round((self.rng.randint(0, 1) - 0.5) * 2) * self.rng.randint(1, 240 // FPS),
                self.rng.randint(1, 480 // FPS)
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

            return Bullet(b_x, b_y, BULLET_SIZE, v_x, v_y)

    def is_colliding(self, r1, r2):
        if (r1.x == r2.x or \
            (r1.x < r2.x and r1.x + r1.width > r2.x) or \
            (r1.x > r2.x and r2.x + r2.width > r1.x)) and \
            (r1.y == r2.y or \
            (r1.y < r2.y and r1.y + r1.height > r2.y) or \
            (r1.y > r2.y and r2.y + r2.height > r1.y)): return True
        return False

    def get_screen(self):
        # creates 2D tensor (32x32) indicating location of bullets
        # first dimension indicates bullets (0 = no bullet, 1 = bullet)
        # second dimension indicates bounds (0 = out of bounds, 1 = in bounds)
        # a third of the dimensions of the screen around the player is used (therefore two thirds of the screen is used)
        # centre (top-left corner of (16, 16)) is player
        s = torch.zeros(1, 32, 32).to(self.device)
        p_x = self.player.pos.x + self.player.pos.width / 2
        p_y = self.player.pos.y + self.player.pos.height / 2
        
        # first dimension
        for i in range(4, len(self.bullets)):
            # for bullets as well
            b_x = self.bullets[i].pos.x + self.bullets[i].pos.width / 2
            b_y = self.bullets[i].pos.y + self.bullets[i].pos.height / 2
            if b_x - p_x >= WIDTH / -3 and b_x - p_x < WIDTH / 3 and \
                b_y - p_y >= HEIGHT / -3 and b_y - p_y < HEIGHT / 3:
                x = math.floor((((b_x - p_x) / (WIDTH / 3)) + 1) * 16)
                y = math.floor((((b_y - p_y) / (HEIGHT / 3)) + 1) * 16)
                s[0][y][x] = 1

        # barrier (previous for loop for bullets only draws one pixel, while barrier uses many pixels)
        # -1 is used in if as barrier is drawn to the side of l_x, r_x, l_y, and r_y, not at that location
        if BULLET_SIZE - 1 - p_x >= WIDTH / -3 and BULLET_SIZE - 1 - p_x < WIDTH / 3:
            l_x = math.floor((((BULLET_SIZE - p_x) / (WIDTH / 3)) + 1) * 16)
            for x in range(l_x):
                for y in range(32): s[0][y][x] = 1
        if WIDTH - 1 - p_x >= WIDTH / -3 and WIDTH - 1 - p_x < WIDTH / 3:
            r_x = math.floor((((WIDTH - BULLET_SIZE - p_x) / (WIDTH / 3)) + 1) * 16)
            for x in range(r_x, 32):
                for y in range(32): s[0][y][x] = 1
        if BULLET_SIZE - 1 - p_y >= HEIGHT / -3 and BULLET_SIZE - 1 - p_y < HEIGHT / 3:
            l_y = math.floor((((BULLET_SIZE - p_y) / (HEIGHT / 3)) + 1) * 16)
            for y in range(l_y): s[0][y] = torch.ones(1, 32)
        if HEIGHT - 1 - p_y >= HEIGHT / -3 and HEIGHT - 1 - p_y < HEIGHT / 3:
            r_y = math.floor((((HEIGHT - BULLET_SIZE - p_y) / (HEIGHT / 3)) + 1) * 16)
            for y in range(r_y, 32): s[0][y] = torch.ones(1, 32)

        return s
