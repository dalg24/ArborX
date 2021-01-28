#include <iostream>

#include "whatever.hpp"

void bar(Whatever) { std::cout << "BAR\n"; }

int a = (Register::add(bar), 0);
