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

#ifndef ARBORX_DETAILS_BOUNDING_VOLUME_HIERARCHY_LEAF_PERMUTATION_ORDER_HPP
#define ARBORX_DETAILS_BOUNDING_VOLUME_HIERARCHY_LEAF_PERMUTATION_ORDER_HPP

#include <ArborX_DetailsHappyTreeFriends.hpp>

namespace ArborX
{
namespace Details
{
template <class BVH>
struct BoundingVolumeHierarchyLeafPermutation
{
  BVH _bvh;
  KOKKOS_FUNCTION int operator()(int i) const
  {
    return HappyTreeFriends::getLeafPermutationIndex(_bvh, i);
  }
};
template <class BVH>
BoundingVolumeHierarchyLeafPermutation<BVH> leafPermutation(BVH &&bvh)
{
  return { (BVH &&)(bvh); }
}
} // namespace Details
} // namespace ArborX

#endif
