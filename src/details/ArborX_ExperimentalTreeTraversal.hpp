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
};

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

template <class BVH, class Predicates, class Callback, class = void>
struct TreeTraversal
{
  BVH _bvh;
  Predicates _predicates;
  Callback _callback;

  using Access = AccessTraits<Predicates, PredicatesTag>;

  using Node = Details::HappyTreeFriends::node_t<BVH>;
  using MemorySpace = typename BVH::memory_space;
  /*static constexpr*/ int undetermined = -1;
  Kokkos::View<int *, MemorySpace> _labels;
  Kokkos::View<int *, MemorySpace> _parents;
  Kokkos::View<int *, MemorySpace> _neighbors;

  template <class ExecutionSpace>
  TreeTraversal(ExecutionSpace const space, BVH const &bvh,
                Predicates const &predicates)
      : _bvh(bvh)
      , _predicates(predicates)
      , _labels(Kokkos::view_alloc(Kokkos::WithoutInitializing, "labels"),
                2 * _bvh.size() - 1)
      , _parents(Kokkos::view_alloc(Kokkos::WithoutInitializing, "parents)"),
                 2 * _bvh.size() - 1)
      , _neighbors(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighbors)"),
            _bvh.size())
  {
    ARBORX_ASSERT(Access::size(predicates) == _bvh.size());
    // reset internal nodes label before each reduction
    Kokkos::deep_copy(
        space,
        Kokkos::subview(_labels, Kokkos::make_pair(0, (int)_bvh.size() - 1)),
        undetermined);
    // initialize leaf nodes labels
    iota(space,
         Kokkos::subview(
             _labels, Kokkos::make_pair(_bvh.size() - 1, 2 * _bvh.size() - 1)));
    Kokkos::parallel_for("ArborX::Experimental::TreeTraversal",
                         Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1),
                         *this);
    Kokkos::parallel_for("ArborX::Experimental::TreeTraversal::ComputeParents",
                         Kokkos::RangePolicy<ExecutionSpace, ComputeParents>(
                             space, 0, _bvh.size() - 1),
                         *this);
    Kokkos::parallel_for("ArborX::Experimental::TreeTraversal::ReduceLabels",
                         Kokkos::RangePolicy<ExecutionSpace, ReduceLabels>(
                             space, _bvh.size() - 1, 2 * _bvh.size() - 1),
                         *this);
  }

  template <typename Tag = typename Node::Tag>
  KOKKOS_FUNCTION
      std::enable_if_t<std::is_same<Tag, Details::NodeWithTwoChildrenTag>{},
                       int>
      getRightChild(Node const *node) const
  {
    assert(!node->isLeaf());
    return node->right_child;
  }

  template <typename Tag = typename Node::Tag>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<Tag, Details::NodeWithLeftChildAndRopeTag>{}, int>
  getRightChild(Node const *node) const
  {
    assert(!node->isLeaf());
    return Details::HappyTreeFriends::getNodePtr(_bvh, node->left_child)->rope;
  }

  struct ComputeParents
  {
  };
  KOKKOS_FUNCTION void operator()(ComputeParents, int i) const
  {
    Node const *node = Details::HappyTreeFriends::getNodePtr(_bvh, i);
    int left_child_index = node->left_child;
    int right_child_index = getRightChild(node);
    _parents(left_child_index) = i;
    _parents(right_child_index) = i;
  }
  struct ReduceLabels
  {
  };
  KOKKOS_FUNCTION void operator()(ReduceLabels, int i) const
  {
    int const root = 0;
    while (true)
    {
      int const lbl = _labels(i);
      int const par = _parents(i);
      int const par_lbl =
          Kokkos::atomic_compare_exchange(&_labels(par), undetermined, lbl);
      // Terminate first thread and let second one continue.
      // This ensures that every node gets processed only once, and not
      // before both of its children are processed.
      if (par_lbl == undetermined)
        break;
      // Do not reduce further and reset to undetermined if children labels do
      // not match
      if (par_lbl != lbl)
      {
        _labels(par) = undetermined;
        break;
      }
      if (par == root)
        break;
      i = par;
    }
  }

  KOKKOS_FUNCTION void operator()(int query_index) const
  {
    auto const &predicate = Access::get(_predicates, query_index);
    auto const distance =
        [geometry = getGeometry(predicate), bvh = _bvh](Node const *node)
    {
      using Details::distance;
      return distance(geometry,
                      Details::HappyTreeFriends::getBoundingVolume(bvh, node));
    };
    auto radius = KokkosExt::ArithmeticTraits::infinity<float>::value;
    int index = undetermined;
    int const leaf_nodes_shift = _bvh.size() - 1;
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
        if (node->isLeaf())
        {
        }
        else
        {
          int const left_child_index = node->left_child;
          int const right_child_index = getRightChild(node);
          Node const *left_child_node =
              Details::HappyTreeFriends::getNodePtr(_bvh, left_child_index);
          Node const *right_child_node =
              Details::HappyTreeFriends::getNodePtr(_bvh, right_child_index);
          float const left_child_distance = distance(left_child_node);
          float const right_child_distance = distance(right_child_node);
          bool const left_child_node_is_leaf = left_child_node->isLeaf();
          bool const right_child_node_is_leaf = right_child_node->isLeaf();
          auto const left_child_label = _labels(
              left_child_node_is_leaf ? left_child_index + leaf_nodes_shift
                                      : left_child_index);
          auto const right_child_label = _labels(
              right_child_node_is_leaf ? right_child_index + leaf_nodes_shift
                                       : right_child_index);
          if (left_child_label != query_label && left_child_distance < radius)
          {
            if (left_child_node_is_leaf)
            {
              index = left_child_index;
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
              radius = right_child_distance;
            }
            else
            {
              stack.emplace(right_child_index, right_child_distance);
            }
          }
        }
      }
    }
    KOKKOS_ASSERT(index != undetermined);
    _neighbors(query_index) = index;
  }
};

template <class ExecutionSpace, class BVH, class Predicates, class Callback>
void traverse(ExecutionSpace const &space, BVH const bvh,
              Predicates const &predicates, Callback const &callback)
{
  TreeTraversal<BVH, Predicates, Callback>(space, bvh, predicates);
}

} // namespace Experimental
} // namespace ArborX

#endif // ARBORX_EXPERIMENTAL_TREE_TRAVERSAL_HPP