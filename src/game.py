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

    def __init__(self):
        self.bullets = [
            Bullet(
                random.randint(0, screenW - 11),
                0,
                random.randint(-4, 4),
                random.randint(0, 6),
                12
            ) for i in range(64)
        ]

        self.player = Player(math.floor(screenW / 2), screenH - 64, 24)

    def Update(self):
        for i in range(len(self.bullets)): 
            self.bullets[i].Update()
            if self.bullets[i].x <= self.bullets[i].size * -1 or \
                self.bullets[i].x >= screenW or \
                self.bullets[i].y <= self.bullets[i].size * -1 or \
                self.bullets[i].y >= screenH:
                    self.bullets[i] = Bullet(
                        random.randint(0, screenW - 1),
                        0,
                        random.randint(-4, 4),
                        random.randint(0, 8),
                        12
                    )
        self.player.Update()

    def Draw(self):
        clear_background(BLACK)
        for i in range(len(self.bullets)): self.bullets[i].Draw()
        self.player.Draw()
