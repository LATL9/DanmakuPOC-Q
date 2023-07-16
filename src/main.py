from common import *

from game import *

from pyray import *

def main():
    # Initialization
    init_window(WIDTH, HEIGHT, "DanmakuPRC")
    set_target_fps(60)

    game = Game()

    # Main game loop
    while not window_should_close():
        # Update
        game.Update()
        if is_key_pressed(KEY_R): print(game.Reset())

        # Draw
        begin_drawing()

        game.Draw()
        draw_fps(8, 8)

        end_drawing()

    # De-Initialization
    close_window()

    return 0

main()
