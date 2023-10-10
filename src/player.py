from common import *

from pyray import *

class Player:
    def __init__(self, x, y, size):
        self.pos = Rectangle(x, y, size, size)

    def copy(self):
        return Player(self.pos.x, self.pos.y, self.pos.width)

    # keys: 0 = up, 1 = down, 2 = left, 3 = right
    def Update(self, keys):
        if keys[0]: self.pos.y -= min(self.pos.y, 400 // GAME_FPS)
        if keys[1]: self.pos.y += min(HEIGHT - self.pos.y - self.pos.height, 400 // GAME_FPS)
        if keys[2]: self.pos.x -= min(self.pos.x, 400 // GAME_FPS)
        if keys[3]: self.pos.x += min(WIDTH - self.pos.x - self.pos.width, 400 // GAME_FPS)

    def Draw(self):
        draw_rectangle_rec(Rectangle(
            self.pos.x - round(self.pos.width * ((TOUCH_SIZE - 1) // 2)),
            self.pos.y - round(self.pos.height * ((TOUCH_SIZE - 1) // 2)),
            self.pos.width * TOUCH_SIZE,
            self.pos.height * TOUCH_SIZE),
            DARKGRAY
        )
        draw_rectangle_rec(Rectangle(
            self.pos.x - round(self.pos.width * ((GRAZE_SIZE - 1) // 2)),
            self.pos.y - round(self.pos.height * ((GRAZE_SIZE - 1) // 2)),
            self.pos.width * GRAZE_SIZE,
            self.pos.height * GRAZE_SIZE),
            LIGHTGRAY
        )
        draw_rectangle_rec(self.pos, WHITE)
