/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_DETAILS_TREE_TRAVERSAL_HPP
#define ARBORX_DETAILS_TREE_TRAVERSAL_HPP

#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsConcepts.hpp>
#include <ArborX_DetailsNode.hpp>
#include <ArborX_DetailsPriorityQueue.hpp>
#include <ArborX_DetailsStack.hpp>
#include <ArborX_Exception.hpp>
#include <ArborX_Predicates.hpp>

#include <Kokkos_TaskScheduler.hpp>

namespace ArborX
{

template <typename DeviceType>
class BoundingVolumeHierarchy;

namespace Details
{
template <typename DeviceType>
struct TreeTraversal
{
public:
  using ExecutionSpace = typename DeviceType::execution_space;

  template <typename Predicate, typename... Args>
  KOKKOS_INLINE_FUNCTION static int
  query(BoundingVolumeHierarchy<DeviceType> const &bvh, Predicate const &pred,
        Args &&... args)
  {
    using Tag = typename Predicate::Tag;
    return queryDispatch(Tag{}, bvh, pred, std::forward<Args>(args)...);
  }

  template <typename Predicates, typename OutputFunctors>
  struct BatchedVisitBoundingVolumeHierarchyNode
  {
    using task_type =
        BatchedVisitBoundingVolumeHierarchyNode<Predicates, OutputFunctors>;
    using value_type = void;
    BoundingVolumeHierarchy<DeviceType> const _bvh;
    Node const *_node;
    Predicates const _predicates;
    OutputFunctors const _output_functors;
    int const _count;
    template <typename TeamMember>
    KOKKOS_INLINE_FUNCTION void operator()(TeamMember &member)
    {
      if (_node->isLeaf())
      {
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, 0, _count), [&](int i) {
              _output_functors[i](_node->getLeafPermutationIndex());
            });
      }
      else
      {
        for (Node const *child :
             {_node->children.first, _node->children.second})
        {
          int cnt = 0;
          Predicates predicates;
          OutputFunctors output_functors;
          Kokkos::parallel_scan<int>(
              Kokkos::TeamThreadRange(member, 0, _count),
              [&](int i, int &, bool is_final) {
                if (is_final && _predicates[i](_bvh.getBoundingVolume(child)))
                {
                  int const pos = Kokkos::atomic_fetch_add(&cnt, 1);
                  predicates[pos] = _predicates[i];
                  output_functors[pos] = _output_functors[i];
                }
              });
          member.team_barrier();
          if (cnt > 0 && member.team_rank() == 0)
          {
            Kokkos::task_spawn(
                Kokkos::TaskTeam(member.scheduler()),
                task_type{_bvh, child, predicates, output_functors, cnt});
          }
        }
      }
    }
  };

  template <typename Predicates>
  static void batchedSpatialQueryExperimental(
      BoundingVolumeHierarchy<DeviceType> const &bvh,
      Predicates const &predicates)
  {
    struct OutputFunctor
    {
      int i;
      KOKKOS_FUNCTION void operator()(int j) const
      {
        printf("trouve %d %d\n", i, j);
      }
    };
    constexpr int N = 10;
    using output_functor_type = Kokkos::Array<OutputFunctor, N>;
    using Predicate = decay_result_of_get_t<
        Traits::Access<Predicates, Traits::PredicatesTag>>;
    using predicate_type = Kokkos::Array<Predicate, N>;
    using execution_space = typename DeviceType::execution_space;
    using memory_space = typename DeviceType::memory_space;
    using task_type =
        BatchedVisitBoundingVolumeHierarchyNode<predicate_type,
                                                output_functor_type>;
    // using execution_space = typename DeviceType::execution_space;
    using scheduler_type = Kokkos::TaskScheduler<execution_space>;
    // using memory_space = typename scheduler_type::memory_space;
    using memory_pool = typename scheduler_type::memory_pool;
    output_functor_type output_functors;
    predicate_type preds;
    // Kokkos::parallel_for(
    //    Kokkos::RangePolicy<execution_space>(0, N), KOKKOS_LAMBDA(int i) {
    for (int i = 0; i < N; ++i)
    {
      output_functors[i] = OutputFunctor{i};
      preds[i] = predicates(i);
    }
    //    });

    size_t const estimate_required_memory = N * N * 8 * sizeof(task_type);

    scheduler_type scheduler{
        memory_pool{memory_space{}, estimate_required_memory}};

    auto ignore = Kokkos::host_spawn(
        Kokkos::TaskSingle(scheduler),
        task_type{bvh, bvh.getRoot(), preds, output_functors, N});
    (void)ignore;
    Kokkos::wait(scheduler);
  }

  template <typename Predicate, typename OutputFunctor>
  struct VisitBoundingVolumeHierarchyNode
  {
    using task_type =
        VisitBoundingVolumeHierarchyNode<Predicate, OutputFunctor>;
    using value_type = void;
    BoundingVolumeHierarchy<DeviceType> const _bvh;
    Node const *_node;
    Predicate const _predicate;
    OutputFunctor const _output_functor;
    template <typename TeamMember>
    KOKKOS_INLINE_FUNCTION void operator()(TeamMember &member)
    {
      if (_node->isLeaf())
      {
        _output_functor(_node->getLeafPermutationIndex());
      }
      else
      {
        for (Node const *child :
             {_node->children.first, _node->children.second})
        {
          if (_predicate(_bvh.getBoundingVolume(child)))
          {
            Kokkos::task_spawn(
                Kokkos::TaskSingle(member.scheduler()),
                task_type{_bvh, child, _predicate, _output_functor});
          }
        }
      }
    }
  };

  template <typename Predicate, typename Insert>
  static void
  spatialQueryExperimental(BoundingVolumeHierarchy<DeviceType> const &bvh,
                           Predicate const &predicate, Insert const &insert)
  {
    using task_type = VisitBoundingVolumeHierarchyNode<Predicate, Insert>;
    using execution_space = typename DeviceType::execution_space;
    using scheduler_type = Kokkos::TaskScheduler<execution_space>;
    using memory_space = typename scheduler_type::memory_space;
    using memory_pool = typename scheduler_type::memory_pool;

    size_t const estimate_required_memory = 8 * sizeof(task_type);

    scheduler_type scheduler{
        memory_pool{memory_space{}, estimate_required_memory}};

    auto ignore =
        Kokkos::host_spawn(Kokkos::TaskSingle(scheduler),
                           task_type{bvh, bvh.getRoot(), predicate, insert});
    (void)ignore;
    Kokkos::wait(scheduler);
  }

  // There are two (related) families of search: one using a spatial predicate
  // and one using nearest neighbours query (see boost::geometry::queries
  // documentation).
  template <typename Predicate, typename Insert>
  KOKKOS_FUNCTION static int
  spatialQuery(BoundingVolumeHierarchy<DeviceType> const &bvh,
               Predicate const &predicate, Insert const &insert)
  {
    if (bvh.empty())
      return 0;

    if (bvh.size() == 1)
    {
      if (predicate(bvh.getBoundingVolume(bvh.getRoot())))
      {
        insert(0);
        return 1;
      }
      else
        return 0;
    }

    Stack<Node const *> stack;

    stack.emplace(bvh.getRoot());
    int count = 0;

    while (!stack.empty())
    {
      Node const *node = stack.top();
      stack.pop();

      if (node->isLeaf())
      {
        insert(node->getLeafPermutationIndex());
        count++;
      }
      else
      {
        for (Node const *child : {node->children.first, node->children.second})
        {
          if (predicate(bvh.getBoundingVolume(child)))
          {
            stack.push(child);
          }
        }
      }
    }
    return count;
  }

  // query k nearest neighbours
  template <typename Distance, typename Insert, typename Buffer>
  KOKKOS_FUNCTION static int
  nearestQuery(BoundingVolumeHierarchy<DeviceType> const &bvh,
               Distance const &distance, std::size_t k, Insert const &insert,
               Buffer const &buffer)
  {
    if (bvh.empty() || k < 1)
      return 0;

    if (bvh.size() == 1)
    {
      insert(0, distance(bvh.getRoot()));
      return 1;
    }

    // Nodes with a distance that exceed that radius can safely be
    // discarded. Initialize the radius to infinity and tighten it once k
    // neighbors have been found.
    double radius = KokkosExt::ArithmeticTraits::infinity<double>::value;

    using PairIndexDistance = Kokkos::pair<int, double>;
    static_assert(
        std::is_same<typename Buffer::value_type, PairIndexDistance>::value,
        "Type of the elements stored in the buffer passed as argument to "
        "TreeTraversal::nearestQuery is not right");
    struct CompareDistance
    {
      KOKKOS_INLINE_FUNCTION bool operator()(PairIndexDistance const &lhs,
                                             PairIndexDistance const &rhs) const
      {
        return lhs.second < rhs.second;
      }
    };
    // Use a priority queue for convenience to store the results and
    // preserve the heap structure internally at all time.  There is no
    // memory allocation, elements are stored in the buffer passed as an
    // argument. The farthest leaf node is on top.
    assert(k == buffer.size());
    PriorityQueue<PairIndexDistance, CompareDistance,
                  UnmanagedStaticVector<PairIndexDistance>>
        heap(UnmanagedStaticVector<PairIndexDistance>(buffer.data(),
                                                      buffer.size()));

    using PairNodePtrDistance = Kokkos::pair<Node const *, double>;
    Stack<PairNodePtrDistance> stack;
    // Do not bother computing the distance to the root node since it is
    // immediately popped out of the stack and processed.
    stack.emplace(bvh.getRoot(), 0.);

    while (!stack.empty())
    {
      Node const *node = stack.top().first;
      double const node_distance = stack.top().second;
      stack.pop();

      if (node_distance < radius)
      {
        if (node->isLeaf())
        {
          int const leaf_index = node->getLeafPermutationIndex();
          double const leaf_distance = node_distance;
          if (heap.size() < k)
          {
            // Insert leaf node and update radius if it was the kth
            // one.
            heap.push(Kokkos::make_pair(leaf_index, leaf_distance));
            if (heap.size() == k)
              radius = heap.top().second;
          }
          else
          {
            // Replace top element in the heap and update radius.
            heap.popPush(Kokkos::make_pair(leaf_index, leaf_distance));
            radius = heap.top().second;
          }
        }
        else
        {
          // Insert children into the stack and make sure that the
          // closest one ends on top.
          Node const *left_child = node->children.first;
          Node const *right_child = node->children.second;
          double const left_child_distance = distance(left_child);
          double const right_child_distance = distance(right_child);
          if (left_child_distance < right_child_distance)
          {
            // NOTE not really sure why but it performed better with
            // the conditional insertion on the device and without
            // it on the host (~5% improvement for both)
#if defined(__CUDA_ARCH__)
            if (right_child_distance < radius)
#endif
              stack.emplace(right_child, right_child_distance);
            stack.emplace(left_child, left_child_distance);
          }
          else
          {
#if defined(__CUDA_ARCH__)
            if (left_child_distance < radius)
#endif
              stack.emplace(left_child, left_child_distance);
            stack.emplace(right_child, right_child_distance);
          }
        }
      }
    }
    // Sort the leaf nodes and output the results.
    // NOTE: Do not try this at home.  Messing with the underlying container
    // invalidates the state of the PriorityQueue.
    sortHeap(heap.data(), heap.data() + heap.size(), heap.valueComp());
    for (decltype(heap.size()) i = 0; i < heap.size(); ++i)
    {
      int const leaf_index = (heap.data() + i)->first;
      double const leaf_distance = (heap.data() + i)->second;
      insert(leaf_index, leaf_distance);
    }
    return heap.size();
  }

  struct Deprecated // using a struct to emulate a nested namespace
  {
    // This is the older version of the nearest traversal that uses a
    // priority queue and that was deemed less performant than the newer
    // version with a stack.
    template <typename Distance, typename Insert>
    KOKKOS_FUNCTION static int
    nearestQuery(BoundingVolumeHierarchy<DeviceType> const &bvh,
                 Distance const &distance, std::size_t k, Insert const &insert)
    {
      if (bvh.empty() || k < 1)
        return 0;

      if (bvh.size() == 1)
      {
        insert(0, distance(bvh.getRoot()));
        return 1;
      }

      using PairNodePtrDistance = Kokkos::pair<Node const *, double>;
      struct CompareDistance
      {
        KOKKOS_INLINE_FUNCTION bool
        operator()(PairNodePtrDistance const &lhs,
                   PairNodePtrDistance const &rhs) const
        {
          // Reverse order (larger distance means lower priority)
          return lhs.second > rhs.second;
        }
      };
      PriorityQueue<PairNodePtrDistance, CompareDistance,
                    StaticVector<PairNodePtrDistance, 256>>
          queue;

      // Do not bother computing the distance to the root node since it is
      // immediately popped out of the stack and processed.
      queue.emplace(bvh.getRoot(), 0.);
      decltype(k) count = 0;

      while (!queue.empty() && count < k)
      {
        // Get the node that is on top of the priority list (i.e. the
        // one that is closest to the query point)
        Node const *node = queue.top().first;
        double const node_distance = queue.top().second;

        if (node->isLeaf())
        {
          queue.pop();
          insert(node->getLeafPermutationIndex(), node_distance);
          ++count;
        }
        else
        {
          // Insert children into the priority queue
          Node const *left_child = node->children.first;
          Node const *right_child = node->children.second;
          double const left_child_distance = distance(left_child);
          double const right_child_distance = distance(right_child);
          queue.popPush(left_child, left_child_distance);
          queue.emplace(right_child, right_child_distance);
        }
      }
      return count;
    }
  }; // "namespace" Deprecated

  template <typename Predicate, typename Insert>
  KOKKOS_INLINE_FUNCTION static int
  queryDispatch(SpatialPredicateTag,
                BoundingVolumeHierarchy<DeviceType> const &bvh,
                Predicate const &pred, Insert const &insert)
  {
    return spatialQuery(bvh, pred, insert);
  }

  template <typename Predicate, typename Insert, typename Buffer>
  KOKKOS_INLINE_FUNCTION static int queryDispatch(
      NearestPredicateTag, BoundingVolumeHierarchy<DeviceType> const &bvh,
      Predicate const &pred, Insert const &insert, Buffer const &buffer)
  {
    auto const geometry = pred._geometry;
    auto const k = pred._k;
    return nearestQuery(
        bvh,
        [geometry, &bvh](Node const *node) {
          return distance(geometry, bvh.getBoundingVolume(node));
        },
        k, insert, buffer);
  }

  // WARNING Without the buffer argument, the dispatch function uses the
  // deprecated version of the nearest query.
  template <typename Predicate, typename Insert>
  KOKKOS_INLINE_FUNCTION static int
  queryDispatch(NearestPredicateTag,
                BoundingVolumeHierarchy<DeviceType> const &bvh,
                Predicate const &pred, Insert const &insert)
  {
    auto const geometry = pred._geometry;
    auto const k = pred._k;
    return Deprecated::nearestQuery(
        // ^^^^^^^^^^
        bvh,
        [geometry, &bvh](Node const *node) {
          return distance(geometry, bvh.getBoundingVolume(node));
        },
        k, insert);
  }
};

} // namespace Details
} // namespace ArborX

#endif
