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
                   Labels const &labels, int n)
{
  ARBORX_ASSERT((int)labels.size() == 2 * n - 1);
  ARBORX_ASSERT((int)parents.size() == 2 * n - 1);
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

template <class BVH>
struct TreeTraversal
{
  BVH _bvh;

  using Node = Details::HappyTreeFriends::node_t<BVH>;
  using MemorySpace = typename BVH::memory_space;
  /*static constexpr*/ int undetermined = -1;
  Kokkos::View<int *, MemorySpace> _labels;
  Kokkos::View<int *, MemorySpace> _neighbors;

  template <class ExecutionSpace>
  TreeTraversal(ExecutionSpace const space, BVH const &bvh,
                Kokkos::View<int *, MemorySpace> labels,
                Kokkos::View<int *, MemorySpace> neighbors)
      : _bvh(bvh)
      , _labels(labels)
      , _neighbors(neighbors)
  {
    ARBORX_ASSERT(labels.size() == 2 * _bvh.size() - 1);
    ARBORX_ASSERT(neighbors.size() == _bvh.size());
    Kokkos::parallel_for(
        "ArborX::Experimental::TreeTraversal",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _bvh.size()),
        // space, _bvh.size() - 1, 2 * _bvh.size() - 1),
        *this);
  }

  KOKKOS_FUNCTION void operator()(int query_index) const
  {
    int const leaf_nodes_shift = _bvh.size() - 1;
    auto const distance =
        [geometry = Details::HappyTreeFriends::getBoundingVolume(
             _bvh, query_index + leaf_nodes_shift),
         bvh = _bvh](Node const *node)
    {
      using Details::distance;
      return distance(geometry,
                      Details::HappyTreeFriends::getBoundingVolume(bvh, node));
    };
    auto radius = KokkosExt::ArithmeticTraits::infinity<float>::value;
    int index = undetermined;
    auto const query_label = _labels(query_index + leaf_nodes_shift);

    using PairIndexDistance = Kokkos::pair<int, float>;
    using Details::Stack;
    Stack<PairIndexDistance> stack;

    Node const *root = Details::HappyTreeFriends::getRoot(_bvh);
    int const root_index = 0;
    stack.emplace(root_index, distance(root));
    while (!stack.empty())
    {
      int const node_index = stack.top().first;
      float const node_distance = stack.top().second;

      stack.pop();

      if (node_distance < radius)
      {
        Node const *node =
            Details::HappyTreeFriends::getNodePtr(_bvh, node_index);
        assert(!node->isLeaf());
        int const left_child_index =
            Details::HappyTreeFriends::getLeftChild(_bvh, node_index);
        int const right_child_index =
            Details::HappyTreeFriends::getRightChild(_bvh, node_index);
        Node const *left_child_node =
            Details::HappyTreeFriends::getNodePtr(_bvh, left_child_index);
        Node const *right_child_node =
            Details::HappyTreeFriends::getNodePtr(_bvh, right_child_index);
        float const left_child_distance = distance(left_child_node);
        float const right_child_distance = distance(right_child_node);
        bool const left_child_node_is_leaf = left_child_node->isLeaf();
        bool const right_child_node_is_leaf = right_child_node->isLeaf();
        auto const left_child_label = _labels(left_child_index);
        auto const right_child_label = _labels(right_child_index);
        if (left_child_label != query_label && left_child_distance < radius)
        {
          if (left_child_node_is_leaf)
          {
            index = left_child_index;
            index = Details::HappyTreeFriends::getLeafPermutationIndex(
                _bvh, left_child_index);
            radius = left_child_distance;
          }
          else
          {
            stack.emplace(left_child_index, left_child_distance);
          }
        }
        if (right_child_label != query_label && right_child_distance < radius)
        {
          if (right_child_node_is_leaf)
          {
            index = right_child_index;
            index = Details::HappyTreeFriends::getLeafPermutationIndex(
                _bvh, right_child_index);
            radius = right_child_distance;
          }
          else
          {
            stack.emplace(right_child_index, right_child_distance);
          }
        }
      }
    }
    KOKKOS_ASSERT(index != undetermined);
    query_index = Details::HappyTreeFriends::getLeafPermutationIndex(
        _bvh, query_index + leaf_nodes_shift);
    _neighbors(query_index) = index;
  }
};

template <class ExecutionSpace, class BVH, class Labels, class Neighbors>
void traverse(ExecutionSpace const &space, BVH const bvh, Labels const &labels,
              Neighbors const &neighbors)
{
  TreeTraversal<BVH>(space, bvh, labels, neighbors);
}

} // namespace Experimental
} // namespace ArborX

#endif // ARBORX_EXPERIMENTAL_TREE_TRAVERSAL_HPP