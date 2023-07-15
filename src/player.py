from common import *

from pyray import *

class Player:
    x = 0
    y = 0
    size = 48

    def __init__(self, _x, _y, _size):
        self.x = _x
        self.y = _y
        self.size = _size

    def Update(self):
        if (is_key_down(KEY_UP)): self.y -= min(self.y, 12)
        if (is_key_down(KEY_DOWN)): self.y += min(screenH - self.y - self.size, 12)
        if (is_key_down(KEY_LEFT)): self.x -= min(self.x, 12)
        if (is_key_down(KEY_RIGHT)): self.x += min(screenW - self.x - self.size, 12)

    def Draw(self):
        draw_rectangle(self.x, self.y, self.size, self.size, WHITE)
