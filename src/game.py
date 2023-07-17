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
            if get_collision_rec(Rectangle(
                self.player.pos.x - round(self.player.pos.width * 3),
                self.player.pos.y - round(self.player.pos.height * 3),
                self.player.pos.width * 7,
                self.player.pos.height * 7),
                self.bullets[i].pos).width != 0:
                self.colliding[0] = True
                if self.invinsible_count[0] == -1: self.invinsible_count[0] = 0
            if get_collision_rec(Rectangle(
                self.player.pos.x - round(self.player.pos.width * 1.5),
                self.player.pos.y - round(self.player.pos.height * 1.5),
                self.player.pos.width * 4,
                self.player.pos.height * 4),
                self.bullets[i].pos).width != 0:
                self.colliding[1] = True
                if self.invinsible_count[1] == -1: self.invinsible_count[1] = 0
            if get_collision_rec(self.player.pos, self.bullets[i].pos).width != 0:
                self.colliding[2] = True
                if self.invinsible_count[2] == -1: self.invinsible_count[2] = 0
        
        for i in range(len(self.invinsible_count)):
            if self.invinsible_count[i] == 0: self.score -= (i + 1) * 60
        self.score += 1

    def Draw(self):
        clear_background(BLACK)
        self.player.Draw()
        for i in range(len(self.bullets)): self.bullets[i].Draw()
        draw_text(str(self.score), 8, 32, 32, WHITE)
        if self.invinsible_count != -1: draw_text(str(self.invinsible_count), 8, 64, 32, DARKGRAY)

    def get_screen(self):
        x = torch.zeros(2, 33, 33).to(self.device)
        x[0, 16, 16] = 1 # one dimension for player

        # other dimension for bullets
        for b in self.bullets:
            if abs(b.pos.x - self.player.pos.x) <= 200 and \
                abs(b.pos.y - self.player.pos.y) <= 200:
                x[1, math.floor((((b.pos.y - self.player.pos.y) / 200) + 1) * 16), math.floor((((b.pos.x - self.player.pos.x) / 200) + 1) * 16)] = 1
        return x