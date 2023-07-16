from common import *

from pyray import *

def gui():
    # Initialization
    init_window(WIDTH, HEIGHT, "DanmakuPRC")
    set_target_fps(60)
	
    while not window_should_close():
        begin_drawing()
        end_drawing()