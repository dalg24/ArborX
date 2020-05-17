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
#ifndef ARBORX_DETAILS_TREE_TRAVERSAL_HPP
#define ARBORX_DETAILS_TREE_TRAVERSAL_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsNode.hpp>
#include <ArborX_DetailsPriorityQueue.hpp>
#include <ArborX_DetailsStack.hpp>
#include <ArborX_Exception.hpp>
#include <ArborX_Macros.hpp>
#include <ArborX_Predicates.hpp>

namespace ArborX
{
namespace Details
{

template <typename BVH, typename Predicates, typename Callback, typename Tag>
struct TreeTraversal
{
};

template <typename BVH, typename Predicates, typename Callback>
struct TreeTraversal<BVH, Predicates, Callback, SpatialPredicateTag>
{
  BVH bvh_;
  Predicates predicates_;
  Callback callback_;

  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;

  template <typename ExecutionSpace>
  TreeTraversal(ExecutionSpace const &space, BVH const &bvh,
                Predicates const &predicates, Callback const &callback)
      : bvh_{bvh}
      , predicates_{predicates}
      , callback_{callback}
  {
    if (bvh_.empty())
    {
      // do nothing
    }
    else if (bvh_.size() == 1)
    {
      Kokkos::parallel_for(
          ARBORX_MARK_REGION("BVH:spatial_queries_degenerated_one_leaf_tree"),
          Kokkos::RangePolicy<ExecutionSpace, OneLeafTree>(
              space, 0, Access::size(predicates)),
          *this);
    }
    else
    {
      experimental_v2(space);
    }
  }

  struct OneLeafTree
  {
  };

  KOKKOS_FUNCTION void operator()(OneLeafTree, int queryIndex) const
  {
    auto const &predicate = Access::get(predicates_, queryIndex);

    if (predicate(bvh_.getBoundingVolume(bvh_.getRoot())))
    {
      callback_(queryIndex, 0);
    }
  }

  KOKKOS_FUNCTION void operator()(int queryIndex) const
  {
    auto const &predicate = Access::get(predicates_, queryIndex);

    Stack<Node const *> stack;

    stack.emplace(bvh_.getRoot());

    while (!stack.empty())
    {
      Node const *node = stack.top();
      stack.pop();

      if (node->isLeaf())
      {
        callback_(queryIndex, node->getLeafPermutationIndex());
      }
      else
      {
        for (Node const *child : {bvh_.getNodePtr(node->children.first),
                                  bvh_.getNodePtr(node->children.second)})
        {
          if (predicate(bvh_.getBoundingVolume(child)))
          {
            stack.push(child);
          }
        }
      }
    }
  }

  struct VisitNode
  {
    using Predicate = typename Traits::Helper<Access>::type;
    using value_type = void;
    BVH bvh_;
    Node const *node_;
    Predicate predicate_;
    int queryIndex_;
    Callback callback_;
    template <typename TeamMember>
    KOKKOS_FUNCTION void operator()(TeamMember &member) const
    {
      if (node_->isLeaf())
      {
        callback_(queryIndex_, node_->getLeafPermutationIndex());
      }
      else
      {
        for (Node const *child : {bvh_.getNodePtr(node_->children.first),
                                  bvh_.getNodePtr(node_->children.second)})
        {
          if (predicate_(bvh_.getBoundingVolume(child)))
          {
            Kokkos::task_spawn(
                Kokkos::TaskSingle(member.scheduler()),
                VisitNode{bvh_, child, predicate_, queryIndex_, callback_});
          }
        }
      }
    }
  };
  struct VisitRoot
  {
    using value_type = void;
    BVH bvh_;
    Predicates predicates_;
    Callback callback_;
    template <typename TeamMember>
    KOKKOS_FUNCTION void operator()(TeamMember &member) const
    {
      int const n_queries = Access::size(predicates_);
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(member, 0, n_queries), [&](int i) {
            Kokkos::task_spawn(Kokkos::TaskSingle(member.scheduler()),
                               VisitNode{bvh_, bvh_.getRoot(),
                                         Access::get(predicates_, i), i,
                                         callback_});
          });
    }
  };
  template <typename ExecutionSpace>
  void experimental(ExecutionSpace const &)
  {
    using scheduler_type = Kokkos::TaskScheduler<ExecutionSpace>;
    using memory_space = typename scheduler_type::memory_space;
    using memory_pool = typename scheduler_type::memory_pool;
    int const n_queries = Access::size(predicates_);
    if (n_queries == 0)
      return;

    size_t const estimate_required_memory =
        n_queries * 1000 * sizeof(VisitNode) + sizeof(VisitRoot);

    scheduler_type scheduler{
        memory_pool{memory_space{}, estimate_required_memory}};

    auto const &bvh = bvh_;
    auto const &predicates = predicates_;
    auto const &callback = callback_;
    auto ignore = Kokkos::host_spawn(Kokkos::TaskTeam(scheduler),
                                     VisitRoot{bvh, predicates, callback});
    (void)ignore;
    Kokkos::wait(scheduler);
  }

  // static constexpr int N = 32;
  template <int N>
  struct VisitNode_v2
  {
    using Predicate = typename Traits::Helper<Access>::type;
    using value_type = void;
    BVH bvh_;
    Node const *node_;
    Kokkos::Array<Predicate, N> predicates_;
    Kokkos::Array<int, N> queryIndices_;
    Callback callback_;
    int n_active_;
    template <typename TeamMember>
    KOKKOS_FUNCTION void operator()(TeamMember &member)
    {
      auto const rank = member.team_rank();
      // printf("Begin visit node %d %d\n", n_active_, rank);
      if (node_->isLeaf())
      {
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, 0, n_active_), [&](int i) {
              // printf("Found collision %d %d %d\n", rank, i,
              //       node_->getLeafPermutationIndex());
              callback_(queryIndices_[i], node_->getLeafPermutationIndex());
            });
      }
      else
      {
        for (Node const *child : {bvh_.getNodePtr(node_->children.first),
                                  bvh_.getNodePtr(node_->children.second)})
        {
          Kokkos::Array<Predicate, N> predicates;
          Kokkos::Array<int, N> queryIndices;
          int n_active = 0;
          Kokkos::parallel_scan<int>(
              Kokkos::TeamThreadRange(member, 0, n_active_),
              [&](int i, int &, bool is_final) {
                if (is_final && predicates_[i](bvh_.getBoundingVolume(child)))
                {
                  // unspecified launch failure when doing atomic increment
                  // int const pos = Kokkos::atomic_fetch_add(&n_active, 1);
                  int const pos = n_active++;
                  // printf("parallel scan %d %d %d\n", pos, n_active_, rank);
                  predicates[pos] = predicates_[i];
                  queryIndices[pos] = queryIndices_[i];
                }
              });
          member.team_barrier();
          if (n_active > 0 && member.team_rank() == 0)
          {
            Kokkos::task_spawn(Kokkos::TaskTeam(member.scheduler()),
                               VisitNode_v2<N>{bvh_, child, predicates,
                                               queryIndices, callback_,
                                               n_active});
          }
        }
      }
    }
  };
  template <int N>
  struct VisitRoot_v2
  {
    using Predicate = typename Traits::Helper<Access>::type;
    using value_type = void;
    BVH bvh_;
    Predicates predicates_;
    Callback callback_;
    template <typename TeamMember>
    KOKKOS_FUNCTION void operator()(TeamMember &member) const
    {
      using KokkosExt::min;
      int const n_queries = Access::size(predicates_);
      Kokkos::Array<Predicate, N> predicates;
      Kokkos::Array<int, N> queryIndices;
      for (int j = 0; j < n_queries; j += N)
      {
        int const NN = min(N, n_queries - j);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, NN),
                             [&](int i) {
                               predicates[i] = Access::get(predicates_, j + i);
                               queryIndices[i] = j + i;
                             });
        member.team_barrier();
        if (member.team_rank() == 0)
        {
          Kokkos::task_spawn(Kokkos::TaskTeam(member.scheduler()),
                             VisitNode_v2<N>{bvh_, bvh_.getRoot(), predicates,
                                             queryIndices, callback_, NN});
        }
      }
    }
  };
  template <typename ExecutionSpace>
  void experimental_v2(ExecutionSpace const &)
  {
    using scheduler_type = Kokkos::TaskScheduler<ExecutionSpace>;
    using memory_space = typename scheduler_type::memory_space;
    using memory_pool = typename scheduler_type::memory_pool;
    int const n_queries = Access::size(predicates_);
    if (n_queries == 0)
      return;
    constexpr int N = 2;

    size_t const estimate_required_memory =
        n_queries * 1000 * sizeof(VisitNode_v2<N>) + sizeof(VisitRoot_v2<N>);

    scheduler_type scheduler{
        memory_pool{memory_space{}, estimate_required_memory}};

    auto const &bvh = bvh_;
    auto const &predicates = predicates_;
    auto const &callback = callback_;
    auto ignore =
        Kokkos::host_spawn(Kokkos::TaskTeam(scheduler),
                           VisitRoot_v2<N>{bvh, predicates, callback});
    (void)ignore;
    Kokkos::wait(scheduler);
  }
};

// NOTE using tuple as a workaround
// Error: class template partial specialization contains a template parameter
// that cannot be deduced
template <typename BVH, typename Predicates, typename Callback>
struct TreeTraversal<BVH, Predicates, Callback, NearestPredicateTag>
{
  using MemorySpace = typename BVH::memory_space;

  BVH bvh_;
  Predicates predicates_;
  Callback callback_;

  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;

  using Buffer = Kokkos::View<Kokkos::pair<int, float> *, MemorySpace>;
  using Offset = Kokkos::View<int *, MemorySpace>;
  struct BufferProvider
  {
    Buffer buffer_;
    Offset offset_;

    KOKKOS_FUNCTION auto operator()(int i) const
    {
      auto const *offset_ptr = &offset_(i);
      return Kokkos::subview(buffer_,
                             Kokkos::make_pair(*offset_ptr, *(offset_ptr + 1)));
    }
  };

  BufferProvider buffer_;

  template <typename ExecutionSpace>
  void allocateBuffer(ExecutionSpace const &space)
  {
    using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
    auto const n_queries = Access::size(predicates_);

    Offset offset(Kokkos::ViewAllocateWithoutInitializing("offset"),
                  n_queries + 1);
    // NOTE workaround to avoid implicit capture of *this
    auto const &predicates = predicates_;
    Kokkos::parallel_for(
        ARBORX_MARK_REGION("scan_queries_for_numbers_of_nearest_neighbors"),
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
        KOKKOS_LAMBDA(int i) { offset(i) = getK(Access::get(predicates, i)); });
    exclusivePrefixSum(space, offset);
    int const buffer_size = lastElement(offset);
    // Allocate buffer over which to perform heap operations in
    // TreeTraversal::nearestQuery() to store nearest leaf nodes found so far.
    // It is not possible to anticipate how much memory to allocate since the
    // number of nearest neighbors k is only known at runtime.

    Buffer buffer(Kokkos::ViewAllocateWithoutInitializing("buffer"),
                  buffer_size);
    buffer_ = BufferProvider{buffer, offset};
  }

  template <typename ExecutionSpace>
  TreeTraversal(ExecutionSpace const &space, BVH const &bvh,
                Predicates const &predicates, Callback const &callback)
      : bvh_{bvh}
      , predicates_{predicates}
      , callback_{callback}
  {
    if (bvh_.empty())
    {
      // do nothing
    }
    else if (bvh_.size() == 1)
    {
      Kokkos::parallel_for(
          ARBORX_MARK_REGION("BVH:nearest_queries_degenerated_one_leaf_tree"),
          Kokkos::RangePolicy<ExecutionSpace, OneLeafTree>(
              space, 0, Access::size(predicates)),
          *this);
    }
    else
    {
      allocateBuffer(space);

      Kokkos::parallel_for(ARBORX_MARK_REGION("BVH:nearest_queries"),
                           Kokkos::RangePolicy<ExecutionSpace>(
                               space, 0, Access::size(predicates)),
                           *this);
    }
  }

  struct OneLeafTree
  {
  };

  KOKKOS_FUNCTION int operator()(OneLeafTree, int queryIndex) const
  {
    auto const &predicate = Access::get(predicates_, queryIndex);
    auto const k = getK(predicate);
    auto const distance = [geometry = getGeometry(predicate),
                           bvh = bvh_](Node const *node) {
      return Details::distance(geometry, bvh.getBoundingVolume(node));
    };

    // NOTE thinking about making this a precondition
    if (k < 1)
      return 0;

    callback_(queryIndex, 0, distance(bvh_.getRoot()));
    return 1;
  }

  KOKKOS_FUNCTION int operator()(int queryIndex) const
  {
    auto const &predicate = Access::get(predicates_, queryIndex);
    auto const k = getK(predicate);
    auto const distance = [geometry = getGeometry(predicate),
                           bvh = bvh_](Node const *node) {
      return Details::distance(geometry, bvh.getBoundingVolume(node));
    };
    auto const buffer = buffer_(queryIndex);

    // NOTE thinking about making this a precondition
    if (k < 1)
      return 0;

    // Nodes with a distance that exceed that radius can safely be
    // discarded. Initialize the radius to infinity and tighten it once k
    // neighbors have been found.
    auto radius = KokkosExt::ArithmeticTraits::infinity<float>::value;

    using PairIndexDistance = Kokkos::pair<int, float>;
    static_assert(
        std::is_same<typename decltype(buffer)::value_type,
                     PairIndexDistance>::value,
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
    assert(k == (int)buffer.size());
    PriorityQueue<PairIndexDistance, CompareDistance,
                  UnmanagedStaticVector<PairIndexDistance>>
        heap(UnmanagedStaticVector<PairIndexDistance>(buffer.data(),
                                                      buffer.size()));

    using PairNodePtrDistance = Kokkos::pair<Node const *, float>;
    Stack<PairNodePtrDistance> stack;
    // Do not bother computing the distance to the root node since it is
    // immediately popped out of the stack and processed.
    stack.emplace(bvh_.getRoot(), 0.);

    while (!stack.empty())
    {
      Node const *node = stack.top().first;
      auto const node_distance = stack.top().second;
      stack.pop();

      if (node_distance < radius)
      {
        if (node->isLeaf())
        {
          int const leaf_index = node->getLeafPermutationIndex();
          auto const leaf_distance = node_distance;
          if ((int)heap.size() < k)
          {
            // Insert leaf node and update radius if it was the kth
            // one.
            heap.push(Kokkos::make_pair(leaf_index, leaf_distance));
            if ((int)heap.size() == k)
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
          Node const *left_child = bvh_.getNodePtr(node->children.first);
          Node const *right_child = bvh_.getNodePtr(node->children.second);
          auto const left_child_distance = distance(left_child);
          auto const right_child_distance = distance(right_child);
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
      auto const leaf_distance = (heap.data() + i)->second;
      callback_(queryIndex, leaf_index, leaf_distance);
    }
    return heap.size();
  }

  struct Deprecated
  {
  };

  // This is the older version of the nearest traversal that uses a priority
  // queue and that was deemed less performant than the newer version with a
  // stack.
  KOKKOS_FUNCTION int operator()(Deprecated, int queryIndex) const
  {
    auto const &predicate = Access::get(predicates_, queryIndex);
    auto const k = getK(predicate);
    auto const distance = [geometry = getGeometry(predicate),
                           bvh = bvh_](Node const *node) {
      return Details::distance(geometry, bvh.getBoundingVolume(node));
    };

    // NOTE thinking about making this a precondition
    if (k < 1)
      return 0;

    using PairNodePtrDistance = Kokkos::pair<Node const *, float>;
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
    queue.emplace(bvh_.getRoot(), 0.);
    decltype(k) count = 0;

    while (!queue.empty() && count < k)
    {
      // Get the node that is on top of the priority list (i.e. the
      // one that is closest to the query point)
      Node const *node = queue.top().first;
      auto const node_distance = queue.top().second;

      if (node->isLeaf())
      {
        queue.pop();
        callback_(queryIndex, node->getLeafPermutationIndex(), node_distance);
        ++count;
      }
      else
      {
        // Insert children into the priority queue
        Node const *left_child = bvh_.getNodePtr(node->children.first);
        Node const *right_child = bvh_.getNodePtr(node->children.second);
        auto const left_child_distance = distance(left_child);
        auto const right_child_distance = distance(right_child);
        queue.popPush(left_child, left_child_distance);
        queue.emplace(right_child, right_child_distance);
      }
    }
    return count;
  }
};

template <typename BVH>
using DeprecatedTreeTraversal = TreeTraversal<BVH, void, void, void>;

template <typename BVH>
struct TreeTraversal<BVH, void, void, void>
{
  // WARNING deprecated will be removed soon
  // still used in TreeVisualization
  template <typename Distance, typename Insert, typename Buffer>
  KOKKOS_FUNCTION static int
  nearestQuery(BVH const &bvh, Distance const &distance, std::size_t k,
               Insert const &insert, Buffer const &buffer)
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
    auto radius = KokkosExt::ArithmeticTraits::infinity<float>::value;

    using PairIndexDistance = Kokkos::pair<int, float>;
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

    using PairNodePtrDistance = Kokkos::pair<Node const *, float>;
    Stack<PairNodePtrDistance> stack;
    // Do not bother computing the distance to the root node since it is
    // immediately popped out of the stack and processed.
    stack.emplace(bvh.getRoot(), 0.);

    while (!stack.empty())
    {
      Node const *node = stack.top().first;
      auto const node_distance = stack.top().second;
      stack.pop();

      if (node_distance < radius)
      {
        if (node->isLeaf())
        {
          int const leaf_index = node->getLeafPermutationIndex();
          auto const leaf_distance = node_distance;
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
          Node const *left_child = bvh.getNodePtr(node->children.first);
          Node const *right_child = bvh.getNodePtr(node->children.second);
          auto const left_child_distance = distance(left_child);
          auto const right_child_distance = distance(right_child);
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
      auto const leaf_distance = (heap.data() + i)->second;
      insert(leaf_index, leaf_distance);
    }
    return heap.size();
  }
};

template <typename ExecutionSpace, typename BVH, typename Predicates,
          typename Callback>
void traverse(ExecutionSpace const &space, BVH const &bvh,
              Predicates const &predicates, Callback const &callback)
{
  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
  using Tag = typename Traits::Helper<Access>::tag;
  TreeTraversal<BVH, Predicates, Callback, Tag>(space, bvh, predicates,
                                                callback);
}

} // namespace Details
} // namespace ArborX

#endif
