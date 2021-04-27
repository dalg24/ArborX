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
  reduce_labels(space, parents, labels);
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, labels);
}
} // namespace Test

#define ARBORX_TEST_REDUCE_LABELS(space, parents, labels, ref)                 \
  BOOST_TEST(Test::reduce_labels(space, parents, labels) == ref,               \
             boost::test_tools::per_element())

namespace Test
{
template <class ExecutionSpace>
auto nearest_neighbor_with_mask(ExecutionSpace const &space,
                                std::vector<ArborX::Point> const &points_host,
                                std::vector<int> const &labels_host)
{
  using MemorySpace = typename ExecutionSpace::memory_space;
  ArborX::BVH<MemorySpace> bvh(
      space, toView<MemorySpace>(points_host, "Test::points"));
  int const n = bvh.size();
  Kokkos::View<int *, MemorySpace> parents(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Test::parents"),
      2 * n - 1);
  Kokkos::View<int *, MemorySpace> labels(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Test::labels"),
      2 * n - 1);
  ArborX::Experimental::init_labels(space, bvh,
                                    toView<MemorySpace>(labels_host), labels);
  ArborX::Experimental::find_parents(space, bvh, parents);
  ArborX::Experimental::reduce_labels(space, parents, labels);
  Kokkos::View<int *, MemorySpace> neighbors(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Test::neighbors"), n);
  ArborX::Experimental::find_nearest_neighbors(space, bvh, labels, neighbors);
  return neighbors;
}
} // namespace Test

#define ARBORX_TEST_NEAREST_NEIGHBOR_WITH_MASK(space, points, labels, ref)     \
  BOOST_TEST(Test::nearest_neighbor_with_mask(space, points, labels) == ref,   \
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
  std::vector<ArborX::Point> points{{0, 0, 0}, {1, 1, 1}, {2, 2, 2}, {3, 3, 3}};
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace space;
  ARBORX_TEST_NEAREST_NEIGHBOR_WITH_MASK(space, points,
                                         (std::vector<int>{0, 0, 0, 1}),
                                         (std::vector<int>{3, 3, 3, 2}));

  ARBORX_TEST_NEAREST_NEIGHBOR_WITH_MASK(space, points,
                                         (std::vector<int>{0, 0, 1, 1}),
                                         (std::vector<int>{2, 2, 1, 1}));

  ARBORX_TEST_NEAREST_NEIGHBOR_WITH_MASK(space, points,
                                         (std::vector<int>{0, 1, 1, 1}),
                                         (std::vector<int>{1, 0, 0, 0}));
}

BOOST_AUTO_TEST_SUITE_END()
