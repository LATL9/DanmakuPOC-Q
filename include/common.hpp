#ifndef COMMON_HPP
#define COMMON_HPP

#include <stdlib.h>
#include <array>

static int screenW = 800;
static int screenH = 540;

template <size_t N, class T>
std::array<T, N> make_array(const T &v)
{
    std::array<T, N> ret;
    ret.fill(v);
    return ret;
}

#endif
