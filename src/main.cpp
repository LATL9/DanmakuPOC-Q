from common import *

from game import *

from pyray import *

def main():
    # Initialization
    init_window(screenW, screenH, "DanmakuPRC")
    set_target_fps(60)

    game = Game()

    # Main game loop
    while not window_should_close():
        # Update
        game.Update()

        # Draw
        begin_drawing()

        game.Draw()

        end_drawing()

    # De-Initialization
    close_window()

    return 0

main()
