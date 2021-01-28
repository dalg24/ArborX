#include <iostream>

#include "whatever.hpp"

void foo(Whatever) { std::cout << "FOO\n"; }

Register b(foo);
