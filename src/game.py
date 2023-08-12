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
        self.bullets = [
            Bullet(
                self.rng.randint(0, WIDTH - 1),
                0,
                12,
                round((self.rng.randint(0, 1) - 0.5) * 2) * self.rng.randint(1, 240 // FPS),
                self.rng.randint(1, 480 // FPS)
            ) for i in range(NUM_BULLETS)
        ]
        self.player = Player(WIDTH // 2, HEIGHT - 64, 24)
        self.score = 0
    
    def End(self): return self.score
    
    def Update(self, keys):
        if TEST_MODEL != -1: self.collides = []

        self.colliding = [False for i in range(len(self.colliding))]
        
        for i in range(len(self.invinsible_count)):
            if self.invinsible_count[i] != -1:
                self.invinsible_count[i] += 1
                if self.invinsible_count[i] == FPS + 1: self.invinsible_count[i] = -1
            
        self.player.Update(keys)
        for i in range(len(self.bullets)): 
            self.bullets[i].Update()
            if self.bullets[i].pos.x <= self.bullets[i].pos.width * -1 or \
                self.bullets[i].pos.x >= WIDTH or \
                self.bullets[i].pos.y <= self.bullets[i].pos.height * -1 or \
                self.bullets[i].pos.y >= HEIGHT:
                    self.bullets[i] = Bullet(
                        self.rng.randint(0, WIDTH - 1),
                        0,
                        12,
                        round((self.rng.randint(0, 1) - 0.5) * 2) * self.rng.randint(1, 240 // FPS),
                        self.rng.randint(1, 480 // FPS)
                    )
            if self.is_colliding(Rectangle(
                self.player.pos.x - round(self.player.pos.width * 3),
                self.player.pos.y - round(self.player.pos.height * 3),
                self.player.pos.width * 7,
                self.player.pos.height * 7),
                self.bullets[i].pos):
                if TEST_MODEL != -1: self.collides.append(i);
                self.colliding[0] = True
                if self.invinsible_count[0] == -1: self.invinsible_count[0] = 0
            if self.is_colliding(Rectangle(
                self.player.pos.x - round(self.player.pos.width * 1.5),
                self.player.pos.y - round(self.player.pos.height * 1.5),
                self.player.pos.width * 4,
                self.player.pos.height * 4),
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
        s = torch.zeros(1, 33, 33).to(self.device)
        
        # centre pixel (16, 16) is player
        for b in self.bullets:
            if abs(b.pos.x - self.player.pos.x) <= WIDTH // 2 and \
                abs(b.pos.y - self.player.pos.y) <= HEIGHT// 2:
                x = math.floor((((b.pos.x - self.player.pos.x) / (WIDTH // 2)) + 1) * 16)
                y = math.floor((((b.pos.y - self.player.pos.y) / (HEIGHT // 2)) + 1) * 16)
                s[0, y, x] = 1

        return s
