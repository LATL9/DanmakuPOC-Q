from common import *

from bullet import *
from player import *

import math
from pyray import *
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Game:
    device = -1
    random = random.seed()
    bullets = []
    player = Player(-1, -1, -1)
    score = 0
    colliding = [False, False, False] # 0 = near player, 1 = grazing player, 2 = touching player
    still_colliding = [False, False, False]
    invinsible_count = [-1, -1, -1] # -1 = not invinsible, 0-59 = invinsible frame (60 is end)
    collides = [] # shows collisions

    def __init__(self, device):
        self.device = device
        self.bullets = [
            Bullet(
                random.randint(0, WIDTH - 11),
                0,
                12,
                round((random.randint(0, 1) - 0.5) * 2) * random.randint(1, 4),
                random.randint(1, 8)
            ) for i in range(NUM_BULLETS)
        ]
        self.player = Player(math.floor(WIDTH / 2), HEIGHT - 64, 24)
        self.score = 0
    
    def End(self): return self.score
    
    def Update(self, keys):
        self.collides = []

        self.colliding = [False for i in range(len(self.colliding))]
        
        for i in range(len(self.invinsible_count)):
            if self.invinsible_count[i] != -1:
                self.invinsible_count[i] += 1
                if self.invinsible_count[i] == 61: self.invinsible_count[i] = -1
            
        self.player.Update(keys)
        for i in range(len(self.bullets)): 
            self.bullets[i].Update()
            if self.bullets[i].pos.x <= self.bullets[i].pos.width * -1 or \
                self.bullets[i].pos.x >= WIDTH or \
                self.bullets[i].pos.y <= self.bullets[i].pos.height * -1 or \
                self.bullets[i].pos.y >= HEIGHT:
                    self.bullets[i] = Bullet(
                        random.randint(0, WIDTH - 1),
                        0,
                        12,
                        round((random.randint(0, 1) - 0.5) * 2) * random.randint(1, 4),
                        random.randint(1, 8)
                    )
            if self.is_colliding(Rectangle(
                self.player.pos.x - round(self.player.pos.width * 3),
                self.player.pos.y - round(self.player.pos.height * 3),
                self.player.pos.width * 7,
                self.player.pos.height * 7),
                self.bullets[i].pos):
                self.collides.append(i);
                self.colliding[0] = True
                if self.invinsible_count[0] == -1: self.invinsible_count[0] = 0
            if self.is_colliding(Rectangle(
                self.player.pos.x - round(self.player.pos.width * 1.5),
                self.player.pos.y - round(self.player.pos.height * 1.5),
                self.player.pos.width * 4,
                self.player.pos.height * 4),
                self.bullets[i].pos):
                self.collides.append(i);
                self.colliding[1] = True
                if self.invinsible_count[1] == -1: self.invinsible_count[1] = 0
            if self.is_colliding(self.player.pos, self.bullets[i].pos):
                self.collides.append(i);
                self.colliding[2] = True
                if self.invinsible_count[2] == -1: self.invinsible_count[2] = 0
        
        for i in range(len(self.invinsible_count)):
            if self.invinsible_count[i] == 0: self.score -= (i + 1) * 60
        self.score += 1

    def Draw(self):
        clear_background(BLACK)
        self.player.Draw()
        for i in range(len(self.bullets)): self.bullets[i].Draw()
        for i in range(len(self.collides)): self.bullets[self.collides[i]].Draw(True)
        draw_text(str(self.score), 8, 32, 32, WHITE)
        if self.invinsible_count != -1: draw_text(str(self.invinsible_count), 8, 64, 32, DARKGRAY)

        # minimap
        s = self.get_screen()
        draw_rectangle(0, 0, 256, 256, Color( 128, 128, 128, 128 ))
        draw_rectangle(16 * 8, 16 * 8, 8, 8, Color( 255, 255, 255, 128 ))
        for y in range(32):
            for x in range(32):
                if s[1, y, x] == 1:  draw_rectangle(x * 8, y * 8, 8, 8, Color( 255, 0, 0, 128 ))

    def is_colliding(self, r1, r2):
        if (r1.x == r2.x or \
            (r1.x < r2.x and r1.x + r1.width > r2.x) or \
            (r1.x > r2.x and r2.x + r2.width > r1.x)) and \
            (r1.y == r2.y or \
            (r1.y < r2.y and r1.y + r1.width > r2.y) or \
            (r1.y > r2.y and r2.y + r2.width > r1.y)): return True
        return False

    def get_screen(self):
        # dimension indicates bullets (0 = no bullet, 1 = bullet)
        x = torch.zeros(1, 33, 33).to(self.device)
        
        # centre pixel (16, 16) is player
        for b in self.bullets:
            if abs(b.pos.x - self.player.pos.x) <= 400 and \
                abs(b.pos.y - self.player.pos.y) <= 400:
                x_pos = math.floor((((b.pos.x - self.player.pos.x) / 400) + 1) * 16)
                y_pos = math.floor((((b.pos.y - self.player.pos.y) / 400) + 1) * 16)
                x[0, y_pos, x_pos] = 1

        return x
