#pragma once

#include <functional>
#include <utility>
#include <vector>

struct Whatever
{
};

struct Register
{
  using Func = std::function<void(Whatever)>;
  static std::vector<Func> d_;
  static void add(Func f) { d_.push_back(std::move(f)); }
  Register(Func f) { add(std::move(f)); }
};
