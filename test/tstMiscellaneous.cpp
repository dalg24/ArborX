/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#define BOOST_TEST_MODULE Miscellaneous
#include "boost_ext/TupleComparison.hpp"
#include <boost/test/unit_test.hpp>

#include <array>
#include <string>

#include "VectorOfTuples.hpp"

namespace Details
{

template <typename T, std::size_t N>
struct ArrayTraits<std::array<T, N>>
{
  using array_type = std::array<T, N>;
  using value_type = typename array_type::value_type;
  static std::size_t size(array_type const &) { return N; }
  static value_type const &access(array_type const &a, std::size_t i)
  {
    return a[i];
  }
};

template <typename T>
struct ArrayTraits<std::vector<T>>
{
  using array_type = std::vector<T>;
  using value_type = typename array_type::value_type;
  static std::size_t size(array_type const &v) { return v.size(); }
  static value_type const &access(array_type const &v, std::size_t i)
  {
    return v[i];
  }
};

} // namespace Details

BOOST_AUTO_TEST_SUITE(VectorOfTuples)

template <typename... Arrays>
auto toVectorOfTuples(Arrays &&... x)
{
  return subsetToVectorOfTuples(0, Details::getSizeOfArrays(x...),
                                std::forward<Arrays>(x)...);
}

BOOST_AUTO_TEST_CASE(heterogeneous)
{
  BOOST_TEST(
      (toVectorOfTuples(
          std::array<std::string, 3>{"dordogne", "gironde", "landes"},
          std::array<int, 3>{24, 33, 40})) ==
          (std::vector<std::tuple<std::string, int>>{
              std::make_tuple("dordogne", 24), std::make_tuple("gironde", 33),
              std::make_tuple("landes", 40)}),
      boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(single_argument)
{
  BOOST_TEST((toVectorOfTuples(std::vector<float>{3.14f})) ==
                 (std::vector<std::tuple<float>>{std::make_tuple(3.14f)}),
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(not_properly_sized)
{
  BOOST_CHECK_EXCEPTION(
      toVectorOfTuples(std::vector<int>{1, 2, 3}, std::vector<double>{3.14},
                       std::vector<std::string>{"foo", "bar"}),
      std::invalid_argument, [&](std::exception const &e) {
        std::string const message = e.what();
        bool const message_contains_argument_position =
            message.find("argument 2") != std::string::npos;
        bool const message_shows_size_mismatch =
            message.find("has size 1 != 3") != std::string::npos;
        return message_contains_argument_position &&
               message_shows_size_mismatch;
      });
}

BOOST_AUTO_TEST_SUITE_END()
