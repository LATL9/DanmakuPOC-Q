from common import *

import math
from pyray import *

class Player:
    pos = Rectangle(-1, -1, -1, -1)

    def __init__(self, x, y, size):
        self.pos = Rectangle(x, y, size, size)

    # keys: 0 = up, 1 = down, 2 = left, 3 = right
    def Update(self, keys):
        if keys[0]: self.pos.y -= min(self.pos.y, 8)
        if keys[1]: self.pos.y += min(HEIGHT - self.pos.y - self.pos.height, 8)
        if keys[2]: self.pos.x -= min(self.pos.x, 8)
        if keys[3]: self.pos.x += min(WIDTH - self.pos.x - self.pos.width, 8)

    def Draw(self):
        draw_rectangle_rec(Rectangle(
            self.pos.x - round(self.pos.width * 3),
            self.pos.y - round(self.pos.height * 3),
            self.pos.width * 7,
            self.pos.height * 7),
            DARKGRAY
        )
        draw_rectangle_rec(Rectangle(
            self.pos.x - round(self.pos.width * 1.5),
            self.pos.y - round(self.pos.height * 1.5),
            self.pos.width * 4,
            self.pos.height * 4),
            LIGHTGRAY
        )
        draw_rectangle_rec(self.pos, WHITE)