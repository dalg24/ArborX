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

#ifndef ARBORX_EXPERIMENTAL_TREE_TRAVERSAL_HPP
#define ARBORX_EXPERIMENTAL_TREE_TRAVERSAL_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsHappyTreeFriends.hpp>
#include <ArborX_DetailsNode.hpp>
#include <ArborX_DetailsStack.hpp>
#include <ArborX_DetailsUtils.hpp> // iota

#include <Kokkos_Core.hpp>

namespace ArborX
{
namespace Experimental
{

template <class ExecutionSpace, class Parents, class Labels>
void reduce_labels(ExecutionSpace const &space, Parents const &parents,
                   Labels const &labels)
{
  ARBORX_ASSERT(parents.size() == labels.size());
  int const n = (parents.size() + 1) / 2;
  ARBORX_ASSERT(n >= 2);
  constexpr typename Labels::value_type undetermined = -1;
  constexpr typename Labels::value_type untouched = -2;
  Kokkos::parallel_for(
      "ArborX::Experimental::reset_parent_labels",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n - 1),
      KOKKOS_LAMBDA(int i) { labels(i) = untouched; });
  Kokkos::parallel_for(
      "ArborX::Experimental::reduce_labels",
      Kokkos::RangePolicy<ExecutionSpace>(space, n - 1, 2 * n - 1),
      KOKKOS_LAMBDA(int i)
      {
        assert(labels(i) != undetermined);
        constexpr typename Labels::value_type root = 0;
        while (true)
        {
          int const lbl = labels(i);
          int const par = parents(i);
          int const par_lbl =
              Kokkos::atomic_compare_exchange(&labels(par), untouched, lbl);
          // Terminate first thread and let second one continue.
          // This ensures that every node gets processed only once, and not
          // before both of its children are processed.
          if (par_lbl == untouched)
            break;
          // Do not reduce further and reset to undetermined if children labels
          // do not match
          if (par_lbl != lbl)
          {
            labels(par) = undetermined;
          }
          if (par == root)
            break;
          i = par;
        }
      });
}

template <class ExecutionSpace, class BVH, class Parents>
void find_parents(ExecutionSpace const &space, BVH const &bvh,
                  Parents const &parents)
{
  Kokkos::parallel_for(
      "ArborX::Experimental::tag_children",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, bvh.size() - 1),
      KOKKOS_LAMBDA(int i)
      {
        int const left_child_index =
            Details::HappyTreeFriends::getLeftChild(bvh, i);
        int const right_child_index =
            Details::HappyTreeFriends::getRightChild(bvh, i);
        parents(left_child_index) = i;
        parents(right_child_index) = i;
      });
}

template <class ExecutionSpace, class BVH, class LabelsIn, class LabelsOut>
void init_labels(ExecutionSpace const &space, BVH const &bvh,
                 LabelsIn const &in, LabelsOut const &out)
{
  ARBORX_ASSERT(in.size() == bvh.size());
  ARBORX_ASSERT(out.size() == 2 * bvh.size() - 1);
  Kokkos::parallel_for(
      "ArborX::Experimental::init_labels",
      Kokkos::RangePolicy<ExecutionSpace>(space, bvh.size() - 1,
                                          2 * bvh.size() - 1),
      KOKKOS_LAMBDA(int i)
      {
        int const leaf_permutation_index =
            Details::HappyTreeFriends::getLeafPermutationIndex(bvh, i);
        out(i) = in(leaf_permutation_index);
      });
}

template <class ExecutionSpace, class BVH, class Labels, class Neighbors>
void find_nearest_neighbors(ExecutionSpace const &space, BVH const bvh,
                            Labels const &labels, Neighbors const &neighbors)
{
  ARBORX_ASSERT(labels.size() == 2 * bvh.size() - 1);
  ARBORX_ASSERT(neighbors.size() == bvh.size());
  constexpr int undetermined = -1;
  constexpr auto infinity = KokkosExt::ArithmeticTraits::infinity<float>::value;
  Kokkos::parallel_for(
      "ArborX::Experimental::find_nearest_neighbors",
      Kokkos::RangePolicy<ExecutionSpace>(space, bvh.size() - 1,
                                          2 * bvh.size() - 1),
      KOKKOS_LAMBDA(int i)
      {
        auto const distance =
            [geometry = Details::HappyTreeFriends::getBoundingVolume(bvh, i),
             &bvh](int j)
        {
          using Details::distance;
          return distance(geometry,
                          Details::HappyTreeFriends::getBoundingVolume(bvh, j));
        };
        auto const predicate = [label = labels(i), &bvh, labels](int j)
        { return labels(j) != label; };
        auto radius = infinity;
        int nearest = undetermined;

        using PairIndexDistance = Kokkos::pair<int, float>;
        using Details::HappyTreeFriends;
        using Details::Stack;
        Stack<PairIndexDistance> stack;

        int const root = 0;
        stack.emplace(root, distance(root));
        while (!stack.empty())
        {
          int const node_index = stack.top().first;
          float const node_distance = stack.top().second;

          stack.pop();

          if (node_distance < radius)
          {
            KOKKOS_ASSERT(!HappyTreeFriends::isLeaf(bvh, node_index));
            int const left_child_index =
                HappyTreeFriends::getLeftChild(bvh, node_index);
            int const right_child_index =
                HappyTreeFriends::getRightChild(bvh, node_index);
            float const left_child_distance = predicate(left_child_index)
                                                  ? distance(left_child_index)
                                                  : infinity;
            float const right_child_distance = predicate(right_child_index)
                                                   ? distance(right_child_index)
                                                   : infinity;
            if (left_child_distance < radius)
            {
              bool const left_child_node_is_leaf =
                  HappyTreeFriends::isLeaf(bvh, left_child_index);
              if (left_child_node_is_leaf)
              {
                auto const left_child_permutation_index =
                    HappyTreeFriends::getLeafPermutationIndex(bvh,
                                                              left_child_index);
                nearest = left_child_permutation_index;
                radius = left_child_distance;
              }
              else
              {
                stack.emplace(left_child_index, left_child_distance);
              }
            }
            if (right_child_distance < radius)
            {
              bool const right_child_node_is_leaf =
                  HappyTreeFriends::isLeaf(bvh, right_child_index);
              if (right_child_node_is_leaf)
              {
                auto const right_child_permutation_index =
                    HappyTreeFriends::getLeafPermutationIndex(
                        bvh, right_child_index);
                nearest = right_child_permutation_index;
                radius = right_child_distance;
              }
              else
              {
                stack.emplace(right_child_index, right_child_distance);
              }
            }
          }
        }
        KOKKOS_ASSERT(nearest != undetermined);
        auto const permutation_index =
            HappyTreeFriends::getLeafPermutationIndex(bvh, i);
        neighbors(permutation_index) = nearest;
      });
}

} // namespace Experimental
} // namespace ArborX

#endif // ARBORX_EXPERIMENTAL_TREE_TRAVERSAL_HPP