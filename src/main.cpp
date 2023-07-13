#include <stdlib.h>
#include <iostream>
#include "common.hpp"

#include "raylib.h"

int main(void)
{
    // Initialization
    int screenWidth = 800;
    int screenHeight = 540;
    screenW = screenWidth;
    screenH = screenHeight;

    InitWindow(screenW, screenH, "DanmakuPRC");
    SetTargetFPS(60);

    // Main game loop
    while (!WindowShouldClose())
    {
        // Update

        // Draw
        BeginDrawing();

        EndDrawing();
    }

    // De-Initialization
    CloseWindow();

    return 0;
}


