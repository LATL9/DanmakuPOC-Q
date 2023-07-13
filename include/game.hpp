#include "common.hpp"

#include "bullet.hpp"
#include "player.hpp"

#include "raylib.h"

class Game
{
    public:
        Game();

        void Init();
        void Update();
        void Draw();

    private:
        Bullet bullet;
        Player player;
};
