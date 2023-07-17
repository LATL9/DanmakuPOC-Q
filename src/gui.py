from common import *

from game import *
from pyray import *

def gui(device):
    # Initialization
    init_window(WIDTH, HEIGHT, "DanmakuPRC")
    set_target_fps(60)
    g = Game(device)
	
    while not window_should_close():
        g.Update([bool(random.getrandbits(1)) for k in range(4)])
        if is_key_pressed(KEY_A): g.get_screen()
        begin_drawing()
        g.Draw()
        end_drawing()