#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "game.hpp"

Game::Game()
{
    srand(time(NULL));

    player.x = 0;
    player.y = 0;
    player.size = 24;
    bullet.x = rand() % screenW;
    bullet.y = 0;
    bullet.v_x = rand() % 10 - 5;
    bullet.v_y = rand() % 5 * -1;
    bullet.size = 12;
};

void Game::Init() { };

void Game::Update()
{
    bullet.x += bullet.v_x;
    bullet.y += bullet.v_y;

    if (IsKeyDown(KEY_UP)) { player.y -= (player.y < 8) ? 0 : 8; }
    if (IsKeyDown(KEY_DOWN)) { player.y += (player.y > screenH - 8) ? 0 : 8; }
    if (IsKeyDown(KEY_LEFT)) { player.x -= (player.x < 8) ? 0 : 8; }
    if (IsKeyDown(KEY_RIGHT)) { player.x += (player.x > screenW - 8) ? 0 : 8; }
};

void Game::Draw()
{
    ClearBackground(BLACK);
    DrawRectangle(player.x - (player.size - 2), player.y - (player.size - 2), player.size, player.size, WHITE);
    DrawRectangle(bullet.x - (bullet.size / 2), bullet.y - (bullet.size / 2), bullet.size, bullet.size, RED);
};
