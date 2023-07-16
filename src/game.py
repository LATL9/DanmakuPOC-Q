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
                12,
                round((random.randint(0, 1) - 0.5) * 2) * random.randint(1, 4),
                random.randint(1, 8)
            ) for i in range(64)
        ]
        print("asdlsadjalk")

        self.player = Player(math.floor(screenW / 2), screenH - 64, 24)

    def Update(self):
        self.player.Update()
        for i in range(len(self.bullets)): 
            self.bullets[i].Update()
            if self.bullets[i].pos.x <= self.bullets[i].pos.width * -1 or \
                self.bullets[i].pos.x >= screenW or \
                self.bullets[i].pos.y <= self.bullets[i].pos.height * -1 or \
                self.bullets[i].pos.y >= screenH:
                    self.bullets[i] = Bullet(
                        random.randint(0, screenW - 1),
                        0,
                        12,
                        round((random.randint(0, 1) - 0.5) * 2) * random.randint(1, 4),
                        random.randint(1, 8)
                    )

    def Draw(self):
        clear_background(BLACK)
        self.player.Draw()
        for i in range(len(self.bullets)): self.bullets[i].Draw()
