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

#include "ArborX_DistributedSearchTree.hpp"
#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>
#include <random>
#include <vector>

#include <mpi.h>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

struct FirstOctant
{
};

struct NearestToOrigin
{
  int k;
};

namespace ArborX
{
template <>
struct AccessTraits<FirstOctant, PredicatesTag>
{
  KOKKOS_FUNCTION static std::size_t size(FirstOctant) { return 1; }
  KOKKOS_FUNCTION static auto get(FirstOctant, std::size_t)
  {
    return intersects(Box{{{0, 0, 0}}, {{1, 1, 1}}});
  }
  using memory_space = MemorySpace;
};
template <>
struct AccessTraits<NearestToOrigin, PredicatesTag>
{
  KOKKOS_FUNCTION static std::size_t size(NearestToOrigin) { return 1; }
  KOKKOS_FUNCTION static auto get(NearestToOrigin d, std::size_t)
  {
    return nearest(Point{0, 0, 0}, d.k);
  }
  using memory_space = MemorySpace;
};
} // namespace ArborX

struct PairIndexDistance
{
  int index;
  float distance;
};

struct PrintfCallback
{
  int _rank;

  template <typename Predicate, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate, int primitive_index,
                                  OutputFunctor const &out) const
  {
    printf("Found %d from functor\n", primitive_index);
    out({primitive_index, _rank, 3.14});
  }
  template <typename Predicate, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate, int primitive, float distance,
                                  OutputFunctor const &out) const
  {
    printf("Found %d with distance %.3f from functor\n", primitive, distance);
    out({primitive, distance});
  }
};

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {

    int const n = 100;
    std::vector<ArborX::Point> points;
    // Fill vector with random points in [-1, 1]^3
    std::uniform_real_distribution<float> dis{-1., 1.};
    std::default_random_engine gen;
    auto rd = [&]() { return dis(gen); };
    std::generate_n(std::back_inserter(points), n, [&]() {
      return ArborX::Point{rd(), rd(), rd()};
    });

    MPI_Comm comm = MPI_COMM_WORLD;

    ArborX::DistributedSearchTree<MemorySpace> tree{
        comm, ExecutionSpace{},
        Kokkos::create_mirror_view_and_copy(
            MemorySpace{}, Kokkos::View<ArborX::Point *, Kokkos::HostSpace,
                                        Kokkos::MemoryUnmanaged>(
                               points.data(), points.size()))};

    {
      struct TupleIndexRankWhatever
      {
        int index;
        int rank;
        double whatever;
      };
      int rank;
      MPI_Comm_rank(comm, &rank);

      Kokkos::View<TupleIndexRankWhatever *, MemorySpace::device_type> values(
          "values", 0);
      Kokkos::View<int *, MemorySpace::device_type> offsets("offsets", 0);
      tree.query(ExecutionSpace{}, FirstOctant{}, PrintfCallback{rank}, values,
                 offsets);

#if 0
      tree.query(ExecutionSpace{}, FirstOctant{},
                 KOKKOS_LAMBDA(auto /*predicate*/
                               ,
                               int primitive, auto /*output_functor*/) {
                   printf("Found %d from generic lambda\n", primitive);
                 },
                 values, offsets, ranks);
#endif
    }
  }
  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
