/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>

struct Dummy
{
  int count;
};

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

template <typename MemorySpace>
struct Iota
{
  static_assert(Kokkos::is_memory_space_v<MemorySpace>);
  using memory_space = MemorySpace;
  int _n;
};

template <typename MemorySpace>
struct ArborX::AccessTraits<Iota<MemorySpace>>
{
  using Self = Iota<MemorySpace>;

  using memory_space = typename Self::memory_space;
  static KOKKOS_FUNCTION size_t size(Self const &self) { return self._n; }
  static KOKKOS_FUNCTION auto get(Self const &, int i) { return i; }
};

struct DummyIndexableGetter
{
  int count;

  using memory_space = MemorySpace;
  KOKKOS_FUNCTION auto size() const { return count; }
  KOKKOS_FUNCTION auto operator()(int i) const
  {
    return ArborX::Point{(float)i, (float)i, (float)i};
  }
};

template <>
struct ArborX::AccessTraits<Dummy>
{
  using memory_space = MemorySpace;
  using size_type = typename MemorySpace::size_type;

  static KOKKOS_FUNCTION size_type size(Dummy const &d) { return d.count; }
  static KOKKOS_FUNCTION auto get(Dummy const &, size_type i)
  {
    ArborX::Point center{(float)i, (float)i, (float)i};
    return ArborX::intersects(Sphere{center, (float)i});
  }
};

template <typename View,
          typename Enable = std::enable_if_t<Kokkos::is_view_v<View>>>
std::ostream &operator<<(std::ostream &os, View const &view)
{
  auto view_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  std::copy(view_host.data(), view_host.data() + view.size(),
            std::ostream_iterator<typename View::value_type>(std::cout, " "));
  return os;
}

struct SomeCallback
{
  template <class P, class V>
  KOKKOS_FUNCTION void operator()(P, V) const
  {}
};

// ARBORX INTERNALS
template <class Callback, class Out>
struct CallbackWrapper
{
  Callback callback_;
  Out out_;
  KOKKOS_FUNCTION CallbackWrapper(Callback const &callback, Out const &out)
      : callback_(callback)
      , out_(out)
  {}
  template <class Predicate, class Value>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate,
                                  Value const &value) const
  {
    if constexpr (std::is_invocable_v<Out const &, Value const &>)
    {
      out_(value);
    }
    else if constexpr (std::is_invocable_v<Out const &>)
    {
      out_();
    }
    else
    {
      static_assert(std::is_void_v<Out>);
    }
    callback_(predicate, value);
  }
};

template <class BVH, class Predicates, class Callback>
struct Foo
{
  BVH bvh_;
  Predicates predicates_;
  Callback callback_;
  Foo(BVH const &bvh, Predicates const &predicates, Callback const &callback)
      : bvh_(bvh)
      , predicates_(predicates)
      , callback_(callback)
  {}
  template <class OutputFunctor>
  KOKKOS_FUNCTION void operator()(int i, OutputFunctor const &out) const
  {
    ArborX::Details::TreeTraversal traverse(bvh_,
                                            CallbackWrapper(callback_, out));
    traverse(predicates_(i));
  }
};
// END ARBORX INTERNALS

template <class ExecutionSpace, class Functor, class Offsets, class Values>
void theAlgoWithNoName(ExecutionSpace const space, Functor const &fun,
                       Offsets const &offsets, Values &values)
{
  int n = offsets.extent(0) + 1;
  int const max_storage = values.extent(0);
  int total_count;
  Kokkos::parallel_scan(
      Kokkos::RangePolicy(space, 0, n),
      KOKKOS_LAMBDA(int i, int &partial_count, bool is_final) {
        int count = 0;
        desul::scoped_atomic_ref<int, desul::MemoryOrderRelaxed,
                                 desul::MemoryScopeDevice>
            ref{count};
        if (!is_final)
        {
          fun(
              i, KOKKOS_LAMBDA() { ++ref; });

          partial_count += count;
        }
        else
        {
          auto offset_i = offsets[i];
          fun(
              i, KOKKOS_LAMBDA(auto val) {
                auto pos = offset_i + ref++;
                if (pos < max_storage)
                  values[pos] = val;
              });
          partial_count += count;
          offsets[i + 1] = partial_count;
        }
      },
      total_count);
  Kokkos::printf("total count %d\n", total_count);
  if (total_count < max_storage)
  {
    return;
  }
  int restart_index;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy(space, 0, n),
      KOKKOS_LAMBDA(int i, int &partial_max) {
        if (i > partial_max && offsets[i + 1] < max_storage)
        {
          partial_max = i;
        }
      },
      Kokkos::Max<int>{restart_index});
  Kokkos::printf("restart index %d\n", restart_index);
  Kokkos::resize(values, total_count);
  Kokkos::parallel_for(
      Kokkos::RangePolicy{space, restart_index, n}, KOKKOS_LAMBDA(int i) {
        int count = 0;
        desul::scoped_atomic_ref<int, desul::MemoryOrderRelaxed,
                                 desul::MemoryScopeDevice>
            ref{count};
        auto offset_i = offsets[i];
        fun(
            i, KOKKOS_LAMBDA(auto val) {
              auto pos = offset_i + ref++;
              values[pos] = val;
            });
        KOKKOS_ASSERT(offsets[i + 1] == offset_i + count);
      });
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  ExecutionSpace space{};

  int nprimitives = 5;
  int npredicates = 5;

  Iota<MemorySpace> primitives{nprimitives};
  DummyIndexableGetter indexable_getter{nprimitives};
  Dummy predicates{npredicates};

  {
    ArborX::BoundingVolumeHierarchy bvh{space, primitives, indexable_getter};

    Kokkos::View<int *, ExecutionSpace> values("Example::values", 20);
    // Kokkos::View<int *, ExecutionSpace> values("Example::values", 4);
    Kokkos::View<int *, ExecutionSpace> offsets("Example::offsets",
                                                npredicates + 1);

    Foo foo{bvh, ArborX::Details::AccessValues<Dummy>{predicates},
            SomeCallback{}};

    theAlgoWithNoName(space, foo, offsets, values);

    std::cout << "offsets (bvh): " << offsets << std::endl;
    std::cout << "values (bvh): " << values << std::endl;
  }

  return 0;
}

