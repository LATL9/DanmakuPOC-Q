#include <stdlib.h>
#include <iostream>
#include "common.hpp"

#include "game.hpp"

int main(void)
{
    // Initialization
    int screenWidth = 800;
    int screenHeight = 540;
    screenW = screenWidth;
    screenH = screenHeight;

    InitWindow(screenW, screenH, "DanmakuPRC");
    SetTargetFPS(60);

    Game game = {};
    game.Init();

    // Main game loop
    while (!WindowShouldClose())
    {
        // Update
        game.Update();

        // Draw
        BeginDrawing();

        game.Draw();

        EndDrawing();
    }

    // De-Initialization
    CloseWindow();

    return 0;
}


