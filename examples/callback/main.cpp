#include "whatever.hpp"

std::vector<Register::Func> Register::d_;

int main()
{
  for (auto const &f : Register::d_)
  {
    f(Whatever{});
  }
}
