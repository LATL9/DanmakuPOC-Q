from common import *

import math
from pyray import *

class Bullet:
    pos = Rectangle(-1, -1, -1, -1)
    v_x = -1
    v_y = -1

    def __init__(self, x, y, w, h, v_x, v_y):
        self.pos = Rectangle(x, y, w, h)
        self.v_x = v_x
        self.v_y = v_y

    def copy(self):
        return Bullet(self.pos.x, self.pos.y, self.pos.width, self.pos.height, self.v_x, self.v_y)

    def Update(self):
        self.pos.x += self.v_x
        self.pos.y += self.v_y

    def Draw(self, blue=False):
        if blue: draw_rectangle_rec(self.pos, BLUE)
        else: draw_rectangle_rec(self.pos, RED)
