from common import *

from bullet import *
from player import *

import math
from pyray import *
import random

class Game:
    random = random.seed()
    bullets = []
    player = Player(-1, -1, -1)
    score = 0
    colliding = False
    still_colliding = False
    invinsible_count = -1 # -1 = not invinsible, 0-59 = invinsible frame (60 is end)

    def __init__(self):
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

    def Update(self):
        collide = False
        if self.invinsible_count != -1:
            self.invinsible_count += 1
            if self.invinsible_count == 61: self.invinsible_count = -1
            
        self.player.Update()
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
            if self.invinsible_count == -1 and \
                get_collision_rec(self.player.pos, self.bullets[i].pos).width != 0:
                if self.colliding: self.still_colliding = True
                self.colliding = True
                collide = True        
        if not collide: self.colliding = False

        if self.colliding:
            if not self.still_colliding:
                self.score -= 6000
                self.invinsible_count = 0
        else:
            self.still_colliding = False
            self.score += 1
    def Draw(self):
        clear_background(BLACK)
        self.player.Draw()
        for i in range(len(self.bullets)): self.bullets[i].Draw()
        draw_text(str(self.score), 8, 32, 32, WHITE)
        if self.invinsible_count != -1: draw_text(str(self.invinsible_count), 8, 64, 32, DARKGRAY)