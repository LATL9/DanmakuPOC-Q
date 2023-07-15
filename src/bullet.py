from common import *

from pyray import *

class Bullet:
    x = -1
    y = -1
    v_x = -1
    v_y = -1
    size = -1

    def __init__(self, _x, _y, _v_x, _v_y, _size):
        self.x = _x
        self.y = _y
        self.v_x = _v_x
        self.v_y = _v_y
        self.size = _size

    def Update(self):
        self.x += self.v_x
        self.y += self.v_y

    def Draw(self):
        draw_rectangle(self.x, self.y, self.size, self.size, RED)
