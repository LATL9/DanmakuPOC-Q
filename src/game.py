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
    invinsible_count = [-1, -1, -1] # -1 = not invinsible, 0 to (FPS - 1) = invinsible frame (FPS is end)
    if TEST_MODEL != -1: collides = [] # shows collisions (used for demonstration, not in training)

    def __init__(self, device, seed):
        self.device = device
        self.rng = random.Random(seed)
        self.player = Player(WIDTH // 2, HEIGHT - 64, 24)
        self.score = 0

        if BULLET_TYPE == BULLET_HONE:
            self.frame_count = FPS // 2 - 1 # used to fire bullets at a constant rate
        else: self.bullets = [self.new_bullet(BULLET_TYPE) for i in range(NUM_BULLETS)]

    

    def End(self): return self.score
    
    def Update(self, keys):
        keys = [is_key_down(KEY_UP), is_key_down(KEY_DOWN), is_key_down(KEY_LEFT), is_key_down(KEY_RIGHT)] 
        if TEST_MODEL != -1: self.collides = []

        self.colliding = [False for i in range(len(self.colliding))]
        
        for i in range(len(self.invinsible_count)):
            if self.invinsible_count[i] != -1:
                self.invinsible_count[i] += 1
                if self.invinsible_count[i] == FPS + 1: self.invinsible_count[i] = -1
            
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

            if self.is_colliding(Rectangle(
                self.player.pos.x - round(self.player.pos.width * 1.5),
                self.player.pos.y - round(self.player.pos.height * 1.5),
                self.player.pos.width * 4,
                self.player.pos.height * 4),
                self.bullets[i].pos):
                if TEST_MODEL != -1: self.collides.append(i);
                self.colliding[0] = True
                if self.invinsible_count[0] == -1: self.invinsible_count[0] = 0
            if self.is_colliding(Rectangle(
                self.player.pos.x - round(self.player.pos.width * 0.75),
                self.player.pos.y - round(self.player.pos.height * 0.75),
                self.player.pos.width * 2.5,
                self.player.pos.height * 2.5),
                self.bullets[i].pos):
                if TEST_MODEL != -1: self.collides.append(i);
                self.colliding[1] = True
                if self.invinsible_count[1] == -1: self.invinsible_count[1] = 0
            if self.is_colliding(self.player.pos, self.bullets[i].pos):
                if TEST_MODEL != -1: self.collides.append(i);
                self.colliding[2] = True
                if self.invinsible_count[2] == -1: self.invinsible_count[2] = 0
        
        for i in range(len(self.invinsible_count)):
            if self.invinsible_count[i] == 0: self.score -= (i + 1) * FPS

        if BULLET_TYPE == BULLET_HONE:
            self.frame_count += 1
            if self.frame_count == FPS // 2:
                self.frame_count = 0
                self.bullets.append(self.new_bullet(BULLET_HONE))

    def Draw(self):
        clear_background(BLACK)
        self.player.Draw()
        for i in range(len(self.bullets)): self.bullets[i].Draw()
        for i in range(len(self.collides)): self.bullets[self.collides[i]].Draw(True)
        draw_text(str(self.score), 8, 32, 32, WHITE)
        if self.invinsible_count != -1: draw_text(str(self.invinsible_count), 8, 64, 32, DARKGRAY)

        # minimap
        s = self.get_screen()
        draw_rectangle(0, 0, 256, 256, Color( 128, 128, 128, 255 ))
        draw_rectangle(16 * 8, 16 * 8, 8, 8, Color( 255, 255, 255, 255 ))
        for y in range(32):
            for x in range(32):
                if s[0, y, x] == 1:  draw_rectangle(x * 8, y * 8, 8, 8, Color( 255, 0, 0, 255 ))

    def new_bullet(self, b):
        if b == BULLET_RANDOM:
            return Bullet(
                self.rng.randint(0, WIDTH - 1),
                -11,
                12,
                round((self.rng.randint(0, 1) - 0.5) * 2) * self.rng.randint(1, 240 // FPS),
                self.rng.randint(1, 480 // FPS)
            )
        elif b == BULLET_HONE:
            x = self.rng.randint(0, WIDTH - 1)
            # maths means all bullets will move at same speed regardless of direction
            # opp from Pythagoras' theorem not specified as opp = player y - 0 = player y
            adj = self.player.pos.x - x
            hyp = pow(pow(adj, 2) + pow(self.player.pos.y + self.player.pos.height / 2, 2), 0.5)
            return Bullet(
                x,
                -11,
                12,
                adj * BULLET_HONE_SPEED / hyp,
                self.player.pos.y * BULLET_HONE_SPEED / hyp
            )
        return Bullet(-1, -1, 1, -1, -1)

    def is_colliding(self, r1, r2):
        if (r1.x == r2.x or \
            (r1.x < r2.x and r1.x + r1.width > r2.x) or \
            (r1.x > r2.x and r2.x + r2.width > r1.x)) and \
            (r1.y == r2.y or \
            (r1.y < r2.y and r1.y + r1.width > r2.y) or \
            (r1.y > r2.y and r2.y + r2.width > r1.y)): return True
        return False

    def get_screen(self):
        # creates 2D tensor (32x32) indicating location of bullets
        # dimension indicates bullets (0 = no bullet, 1 = bullet)
        # a third of the dimensions of the screen around the player is used (therefore two thirds of the screen is used)
        # centre (top-left corner of (16, 16)) is player
        s = torch.zeros(1, 32, 32).to(self.device)
        p_x = self.player.pos.x + self.player.pos.width / 2
        p_y = self.player.pos.y + self.player.pos.height / 2
        
        for b in self.bullets:
            b_x = b.pos.x + b.pos.width / 2
            b_y = b.pos.y + b.pos.height / 2
            if b_x - p_x >= WIDTH / -3 and b_x - p_x < WIDTH / 3 and \
                b_y - p_y >= HEIGHT / -3 and b_y - p_y < HEIGHT / 3:
                x = math.floor((((b_x - p_x) / (WIDTH / 3)) + 1) * 16)
                y = math.floor((((b_y - p_y) / (HEIGHT / 3)) + 1) * 16)
                s[0, y, x] = 1

        return s
