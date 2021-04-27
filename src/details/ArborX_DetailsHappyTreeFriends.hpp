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

#ifndef ARBORX_DETAILS_HAPPY_TREE_FRIENDS_HPP
#define ARBORX_DETAILS_HAPPY_TREE_FRIENDS_HPP

#include <ArborX_DetailsNode.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_Pair.hpp>

#include <type_traits>
#include <utility> // declval

namespace ArborX
{
namespace Details
{
struct HappyTreeFriends
{
  template <class BVH>
  static KOKKOS_FUNCTION decltype(auto) getNodePtr(BVH const &bvh, int i)
  {
    return bvh.getNodePtr(i);
  }

  template <class BVH>
  static KOKKOS_FUNCTION decltype(auto) getRoot(BVH const &bvh)
  {
    return bvh.getRoot();
  }

  template <class BVH>
  using node_const_pointer_t = decltype(getRoot(std::declval<BVH const &>()));

  template <class BVH>
  using node_t =
      std::remove_const_t<std::remove_pointer_t<node_const_pointer_t<BVH>>>;

  template <class BVH>
  static KOKKOS_FUNCTION decltype(auto)
  getBoundingVolume(BVH const &bvh, node_t<BVH> const *node)
  {
    return bvh.getBoundingVolume(node);
  }

  template <class BVH>
  static decltype(auto) getLeafNodes(BVH const &bvh)
  {
    return bvh.getLeafNodes();
  }

  template <class BVH>
  static KOKKOS_FUNCTION int getLeftChild(BVH const &bvh, int i)
  {
    return bvh.getNodePtr(i)->left_child;
  }

  template <class BVH, std::enable_if_t<std::is_same<
                           typename node_t<BVH>::Tag,
                           Details::NodeWithTwoChildrenTag>::value> * = nullptr>
  static KOKKOS_FUNCTION int getRightChild(BVH const &bvh, int i)
  {
    return bvh.getNodePtr(i)->right_child;
  }

  template <class BVH,
            std::enable_if_t<std::is_same<
                typename node_t<BVH>::Tag,
                Details::NodeWithLeftChildAndRopeTag>::value> * = nullptr>
  static KOKKOS_FUNCTION int getRightChild(BVH const &bvh, int i)
  {
    return bvh.getNodePtr(bvh.getNodePtr(i)->left_child)->rope;
  }

  template <class BVH>
  static KOKKOS_FUNCTION int getLeafPermutationIndex(BVH const &bvh, int i)
  {
    return bvh.getNodePtr(i)->getLeafPermutationIndex();
  }

  template <class BVH>
  static KOKKOS_FUNCTION decltype(auto) getBoundingVolume(BVH const &bvh, int i)
  {
    return bvh.getBoundingVolume(bvh.getNodePtr(i));
  }
};
} // namespace Details
} // namespace ArborX

#endif
