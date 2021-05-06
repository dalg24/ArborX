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
template <class BVH>
struct NearestK
{
  BVH _bvh;
  int _k;
};

template <class BVH, class Permute, class Distances>
struct MaxDistance
{
  BVH _bvh;
  Permute _permute;
  Distances _distances;
  template <class Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate, int i) const
  {
    using Details::distance;
    using Details::HappyTreeFriends;
    using KokkosExt::max;
    int const ii = _permute(i);
    int const jj = getData(predicate);
    auto const &bounding_volume_i =
        HappyTreeFriends::getBoundingVolume(_bvh, ii);
    auto const &bounding_volume_j =
        HappyTreeFriends::getBoundingVolume(_bvh, jj);
    // FIXME using knowledge that one query processed by a single thread
    // must be atomic otherwise.
    // FIXME might want distance entries in permuted order
    int const leaf_nodes_shift = _bvh.size() - 1;
    int const j = jj - leaf_nodes_shift;
    _distances(j) =
        max(_distances(j), distance(bounding_volume_i, bounding_volume_j));
  }
};
} // namespace Experimental
} // namespace ArborX

template <class BVH>
struct ArborX::AccessTraits<ArborX::Experimental::NearestK<BVH>,
                            ArborX::PredicatesTag>
{
  using size_type = typename BVH::size_type;
  using memory_space = typename BVH::memory_space;
  using Self = ArborX::Experimental::NearestK<BVH>;
  static KOKKOS_FUNCTION size_type size(Self const &self)
  {
    return self._bvh.size();
  }
  static KOKKOS_FUNCTION auto get(Self const &self, size_type i)
  {
    using Details::HappyTreeFriends;
    int const leaf_nodes_shift = size(self) - 1;
    int const ii = i + leaf_nodes_shift;
    return attach(
        nearest(HappyTreeFriends::getBoundingVolume(self._bvh, ii), self._k),
        (int)ii);
  }
};

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
          // Do not reduce further and reset to undetermined if children
          // labels do not match
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
        auto const predicate = [label = labels(i), labels](int j)
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

template <class ExecutionSpace, class BVH, class Permute>
void reverse_permutation(ExecutionSpace const &space, BVH const &bvh,
                         Permute const &permute)
{
  Kokkos::parallel_for(
      "ArborX::Experimental::reverse_leaf_permutation",
      Kokkos::RangePolicy<ExecutionSpace>(space, bvh.size() - 1,
                                          2 * bvh.size() - 1),
      KOKKOS_LAMBDA(int i)
      {
        using Details::HappyTreeFriends;
        auto const leaf_permutation_index =
            HappyTreeFriends::getLeafPermutationIndex(bvh, i);
        permute(leaf_permutation_index) = i;
      });
}

template <class ExecutionSpace, class BVH, class Distances>
void find_kth_neighbors(ExecutionSpace const &space, BVH const &bvh, int k,
                        Distances const &distances)
{
  Kokkos::deep_copy(distances, -1);

  using MemorySpace = typename BVH::memory_space;
  Kokkos::View<int *, MemorySpace> permute(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Experimental::permute"),
      bvh.size());
  reverse_permutation(space, bvh, permute);

  // NOTE using k+1 to account for self-collision
  bvh.query(
      space, NearestK<BVH>{bvh, k + 1},
      MaxDistance<BVH, decltype(permute), Distances>{bvh, permute, distances});
}

template <class ExecutionSpace, class Labels>
void merge_labels(ExecutionSpace const &space, Labels const &labels)
{
  int const n = labels.size();
  Kokkos::parallel_for(
      "ArborX::Experimental::merge_labels",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
      KOKKOS_LAMBDA(int i)
      {
        int label_i = labels(i);
        if (i != label_i)
        {
          int ii;
          do
          {
            ii = label_i;
            label_i = labels(ii);
          } while (ii != label_i);
          labels(i) = ii;
        }
      });
}

template <class ExecutionSpace, class Labels, class Components>
void find_components(ExecutionSpace const &space, Labels const &labels,
                     Components const &components)
{
  using MemorySpace = typename ExecutionSpace::memory_space;
  int const n = components.size();
  ARBORX_ASSERT((int)labels.size() == 2 * n - 1);

  Kokkos::View<int *, MemorySpace> offsets(
      "ArborX::Experimental::clusters_offsets", n + 1);
  Kokkos::parallel_for(
      "ArborX::ExecutionSpace::compute_cluster_sizes",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
      KOKKOS_LAMBDA(int i) { Kokkos::atomic_increment(offsets(&labels(i))); });
  exclusivePrefixSum(space, offsets);
  Kokkos::View<int *, MemorySpace> indices(
      "ArborX::Experimental::clusters_indices", lastElement(offsets));

  Kokkos::parallel_scan("ArborX::Experimental::find_components",
                        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                        KOKKOS_LAMBDA(int i, int &update, bool final_pass){

                        });
}

template <class ExecutionSpace, class BVH>
void build_minimum_spanning_tree(ExecutionSpace const &space, BVH const &bvh)
{
  using MemorySpace = typename BVH::memory_space;
  int const n = bvh.size();
  Kokkos::View<int *, MemorySpace> components(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Experimental::components"),
      n);
}

} // namespace Experimental
} // namespace ArborX

#endif // ARBORX_EXPERIMENTAL_TREE_TRAVERSAL_HPP