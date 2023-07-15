from common import *

from bullet import *
from player import *

import math
from pyray import *
import random

class Game:
    random = random.seed()
    bullet = Bullet()
    player = Player()

    def __init__(self):
        self.player.x = 0
        self.player.y = 0
        self.player.size = 24
        self.bullet.x = random.randint(0, screenW - 1)
        self.bullet.y = 0
        self.bullet.v_x = random.randint(-1, 1)
        self.bullet.v_y = random.randint(0, 8)
        self.bullet.size = 12

    def Update(self):
        self.bullet.x += self.bullet.v_x
        self.bullet.y += self.bullet.v_y

        if (is_key_down(KEY_UP)): self.player.y -= min(self.player.y, 12)
        if (is_key_down(KEY_DOWN)): self.player.y += min(screenH - self.player.y - self.player.size, 12)
        if (is_key_down(KEY_LEFT)): self.player.x -= min(self.player.x, 12)
        if (is_key_down(KEY_RIGHT)): self.player.x += min(screenW - self.player.x - self.player.size, 12)

    def Draw(self):
        clear_background(BLACK)
        draw_rectangle(self.player.x, self.player.y, self.player.size, self.player.size, WHITE)
        draw_rectangle(self.bullet.x, self.bullet.y, self.bullet.size, self.bullet.size, RED)
