from common import *

import math
from pyray import *

class Player:
    pos = Rectangle(-1, -1, -1, -1)

    def __init__(self, x, y, size):
        self.pos = Rectangle(x, y, size, size)

    def Update(self):
        if (is_key_down(KEY_UP)): self.pos.y -= min(self.pos.y, 8)
        if (is_key_down(KEY_DOWN)): self.pos.y += min(HEIGHT - self.pos.y - self.pos.height, 8)
        if (is_key_down(KEY_LEFT)): self.pos.x -= min(self.pos.x, 8)
        if (is_key_down(KEY_RIGHT)): self.pos.x += min(WIDTH - self.pos.x - self.pos.width, 8)

    def Draw(self):
        draw_rectangle_rec(Rectangle(
            self.pos.x - round(self.pos.width * 2),
            self.pos.y - round(self.pos.height * 2),
            self.pos.width * 5,
            self.pos.height * 5),
            DARKGRAY
        )
        draw_rectangle_rec(Rectangle(
            self.pos.x - round(self.pos.width * 1),
            self.pos.y - round(self.pos.height * 1),
            self.pos.width * 3,
            self.pos.height * 3),
            LIGHTGRAY
        )
        draw_rectangle_rec(self.pos, WHITE)