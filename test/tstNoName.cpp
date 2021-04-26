/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_ExperimentalTreeTraversal.hpp>
#include <ArborX_LinearBVH.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

template <typename DeviceType, typename T>
auto toView(std::vector<T> const &v, std::string const &lbl = "")
{
  Kokkos::View<T *, DeviceType> view(lbl, v.size());
  Kokkos::deep_copy(view, Kokkos::View<T const *, Kokkos::HostSpace,
                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                              v.data(), v.size()));
  return view;
}

namespace Test
{
template <class ExecutionSpace>
auto reduce_labels(ExecutionSpace const &space,
                   std::vector<int> const &parents_host,
                   std::vector<int> const &labels_host)
{
  using ArborX::Experimental::reduce_labels;
  auto labels = toView<ExecutionSpace>(labels_host, "Test::labels");
  auto parents = toView<ExecutionSpace>(parents_host, "Test::parents");
  reduce_labels(space, parents, labels, (parents.size() + 1) / 2);
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, labels);
}
} // namespace Test

#define ARBORX_TEST_REDUCE_LABELS(space, parents, labels, ref)                 \
  BOOST_TEST(Test::reduce_labels(space, parents, labels) == ref,               \
             boost::test_tools::per_element())

BOOST_AUTO_TEST_SUITE(NoName)

BOOST_AUTO_TEST_CASE_TEMPLATE(reduce_labels, DeviceType, ARBORX_DEVICE_TYPES)
{
  // [0]------------*--------------
  //               / \
  //  ------*----[3] [4]*----------
  //       / \         / \
  //  --*[1] [2]*--   |  [5]----*--
  //   / \     / \    |        / \
  //  |   |   |   |   |   --*[6]  |
  //  |   |   |   |   |    / \    |
  //  0   1   2   3   4   5   6   7
  auto const parents =
      std::vector<int>{-1, 3, 3, 0, 0, 4, 5, 1, 1, 2, 2, 4, 6, 6, 5};
  //                    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
  //                   [0][1][2][3][4][5][6] 0  1  2  3  4  5  6  7

  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace space;

  ARBORX_TEST_REDUCE_LABELS(
      space, parents,
      (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0}),
      (std::vector<int>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

  ARBORX_TEST_REDUCE_LABELS(
      space, parents,
      (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7}),
      (std::vector<int>{7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7}));

  ARBORX_TEST_REDUCE_LABELS(
      space, parents,
      (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7}),
      (std::vector<int>{-1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7}));

  ARBORX_TEST_REDUCE_LABELS(
      space, parents,
      (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 3, 4, 4, 4, 7}),
      (std::vector<int>{-1, 0, -1, -1, -1, -1, 4, 0, 0, 0, 3, 4, 4, 4, 7}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(nearest_neighbor, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ArborX::Point;
  auto points = toView<DeviceType>(
      std::vector<ArborX::Point>{{0, 0, 0}, {1, 1, 1}, {2, 2, 2}, {3, 3, 3}},
      "Test::points");
  auto labels =
      toView<DeviceType>(std::vector<int>{0, 1, 2, 0, 0, 0, 1}, "Test::labels");

  auto predicates = toView<DeviceType>(
      std::vector<decltype(ArborX::nearest(ArborX::Point{}))>{
          ArborX::nearest(ArborX::Point{0, 0, 0}),
          ArborX::nearest(ArborX::Point{1, 1, 1}),
          ArborX::nearest(ArborX::Point{2, 2, 2}),
          ArborX::nearest(ArborX::Point{3, 3, 3})},
      "Test::predicates");

  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  ExecutionSpace space;

  ArborX::BVH<MemorySpace> bvh(space, points);

  int const n = bvh.size();
  Kokkos::View<int *, MemorySpace> parents(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Test::parents"),
      2 * n - 1);
  ArborX::Experimental::find_parents(space, bvh, parents);
  ArborX::Experimental::reduce_labels(space, parents, labels, n);
  Kokkos::View<int *, MemorySpace> neighbors(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Test::neighbors"), n);
  ArborX::Experimental::traverse(space, bvh, predicates, labels, neighbors);
  BOOST_TEST(neighbors == (std::vector<int>{3, 3, 3, 2}),
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_SUITE_END()
