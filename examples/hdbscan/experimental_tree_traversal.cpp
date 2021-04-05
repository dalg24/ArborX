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

#include <ArborX_ExperimentalTreeTraversal.hpp>
#include <ArborX_LinearBVH.hpp>

#include <fstream>
#include <iostream>
#include <vector>

// stolen from Piyush
std::vector<std::vector<double>> readFile(std::ifstream &inFile)
{
  char c;
  int npts;
  int dim;
  inFile >> c;
  inFile >> npts;
  inFile >> dim;
  std::cout << c << " number of pts is " << npts << " and dimension is " << dim
            << "\n";
  std::vector<std::vector<double>> out(npts, std::vector<double>(dim, 0.0));
  for (int i = 0; i < npts; i++)
    for (int j = 0; j < dim; j++)
      inFile >> out[i][j];

  return out;
}

struct VectorOfVectors
{
  std::vector<std::vector<double>> _v;
};

template <>
struct ArborX::AccessTraits<VectorOfVectors, ArborX::PrimitivesTag>
{
  using memory_space = Kokkos::HostSpace;
  using size_type = memory_space::size_type;
  static KOKKOS_FUNCTION size_type size(VectorOfVectors const &v)
  {
    return v._v.size();
  }
  static KOKKOS_FUNCTION auto get(VectorOfVectors const &v, size_type i)
  {
    auto const &p = v._v[i];
    return Point{{p[0], p[1], 0}};
  }
};

template <>
struct ArborX::AccessTraits<VectorOfVectors, ArborX::PredicatesTag>
{
  using memory_space = Kokkos::HostSpace;
  using size_type = memory_space::size_type;
  static KOKKOS_FUNCTION size_type size(VectorOfVectors const &v)
  {
    return v._v.size();
  }
  static KOKKOS_FUNCTION auto get(VectorOfVectors const &v, size_type i)
  {
    auto const &p = v._v[i];
    return nearest(Point{{p[0], p[1], 0}});
  }
};

int main(int argc, char *argv[])
{
  // read the points
  if (argc < 2)
  {
    std::cout << " usage: " << argv[0] << " inputFile\n";
    return EXIT_FAILURE;
  }
  std::cout << "Opening " << argv[1] << "\n";
  std::ifstream infile(argv[1]);

  std::vector<std::vector<double>> points = readFile(infile);

  Kokkos::ScopeGuard exec_env_guard(argc, argv);
  Kokkos::DefaultHostExecutionSpace space;
  ArborX::BVH<Kokkos::HostSpace> bvh(space, VectorOfVectors{points});
  ArborX::Experimental::traverse(space, bvh, VectorOfVectors{points}, 2);

  return EXIT_SUCCESS;
}