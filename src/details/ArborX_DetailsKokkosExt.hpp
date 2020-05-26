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
#ifndef ARBORX_DETAILS_KOKKOS_EXT_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_HPP

#include <ArborX_CudaOccupancy.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>
#include <cfloat>  // DBL_MAX, DBL_EPSILON
#include <cmath>   // isfinite, HUGE_VAL
#include <cstdint> // uint32_t
#include <type_traits>

#if __cplusplus < 201402L
namespace std
{
template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;
} // namespace std
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace KokkosExt
{
template <typename MemorySpace, typename ExecutionSpace, typename = void>
struct is_accessible_from : std::false_type
{
  static_assert(Kokkos::is_memory_space<MemorySpace>::value, "");
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value, "");
};

template <typename MemorySpace, typename ExecutionSpace>
struct is_accessible_from<MemorySpace, ExecutionSpace,
                          typename std::enable_if<Kokkos::SpaceAccessibility<
                              ExecutionSpace, MemorySpace>::accessible>::type>
    : std::true_type
{
};

template <typename View>
struct is_accessible_from_host
    : public is_accessible_from<typename View::memory_space, Kokkos::HostSpace>
{
  static_assert(Kokkos::is_view<View>::value, "");
};

/** Count the number of consecutive leading zero bits in 32 bit integer
 * @param x
 */
KOKKOS_INLINE_FUNCTION
int clz(uint32_t x)
{
#if defined(__CUDA_ARCH__)
  // Note that the __clz() CUDA intrinsic function takes a signed integer
  // as input parameter.  This is fine but would need to be adjusted if
  // we were to change expandBits() and morton3D() to subdivide [0, 1]^3
  // into more 1024^3 bins.
  return __clz(x);
#elif defined(KOKKOS_COMPILER_GNU) || (KOKKOS_COMPILER_CLANG >= 500)
  // According to https://en.wikipedia.org/wiki/Find_first_set
  // Clang 5.X supports the builtin function with the same syntax as GCC
  return (x == 0) ? 32 : __builtin_clz(x);
#else
  if (x == 0)
    return 32;
  // The following is taken from:
  // http://stackoverflow.com/questions/23856596/counting-leading-zeros-in-a-32-bit-unsigned-integer-with-best-algorithm-in-c-pro
  const char debruijn32[32] = {0,  31, 9,  30, 3,  8,  13, 29, 2,  5, 7,
                               21, 12, 24, 28, 19, 1,  10, 4,  14, 6, 22,
                               25, 20, 11, 15, 23, 26, 16, 27, 17, 18};
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return debruijn32[x * 0x076be629 >> 27];
#endif
}

//! Compute the maximum of two values.
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr T const &max(T const &a, T const &b)
{
  return (a > b) ? a : b;
}

//! Compute the minimum of two values.
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr T const &min(T const &a, T const &b)
{
  return (a < b) ? a : b;
}

/**
 * Branchless sign function. Return 1 if @param x is greater than zero, 0 if
 * @param x is zero, and -1 if @param x is less than zero.
 */
template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
KOKKOS_INLINE_FUNCTION int sgn(T x)
{
  return (x > 0) - (x < 0);
}

/** Determine whether the given floating point argument @param x has finite
 * value.
 *
 * NOTE: Clang issues a warning if the std:: namespace is missing and nvcc
 * complains about calling a __host__ function from a __host__ __device__
 * function when it is present.
 */
template <typename T>
KOKKOS_INLINE_FUNCTION bool isFinite(T x)
{
#ifdef __CUDA_ARCH__
  return isfinite(x);
#else
  return std::isfinite(x);
#endif
}

namespace ArithmeticTraits
{

template <typename T>
struct infinity;

template <>
struct infinity<float>
{
  static constexpr float value = HUGE_VALF;
};

template <>
struct infinity<double>
{
  static constexpr double value = HUGE_VAL;
};

template <typename T>
struct max;

template <>
struct max<float>
{
  static constexpr float value = FLT_MAX;
};

template <>
struct max<double>
{
  static constexpr double value = DBL_MAX;
};

template <typename T>
struct epsilon;

template <>
struct epsilon<float>
{
  static constexpr float value = FLT_EPSILON;
};

template <>
struct epsilon<double>
{
  static constexpr double value = DBL_EPSILON;
};

} // namespace ArithmeticTraits

namespace DoNotTryThisAtHome
{

class BlockSize
{
  unsigned const size_;

public:
  explicit constexpr BlockSize(unsigned size) noexcept
      : size_{size}
  {
  }
  constexpr operator unsigned() const noexcept { return size_; }
};

class SharedMemSize
{
  unsigned const size_;

public:
  explicit constexpr SharedMemSize(unsigned size) noexcept
      : size_{size}
  {
  }
  constexpr operator unsigned() const noexcept { return size_; }
};

class Occupancy
{
  int const percent_;

public:
  explicit constexpr Occupancy(int percent) noexcept
      : percent_{percent}
  {
    assert(percent >= 0 && percent <= 100);
  }
  constexpr operator int() const noexcept { return percent_; }
};

template <typename FunctorType, typename ExecPolicy>
class ParallelFor;

template <class FunctorType, class ExecPolicy>
inline std::enable_if_t<
    !std::is_same<typename Kokkos::Impl::FunctorPolicyExecutionSpace<
                      FunctorType, ExecPolicy>::execution_space,
                  Kokkos::Cuda>{}>
parallel_for(const std::string &str, const ExecPolicy &policy,
             const FunctorType &functor, BlockSize, Occupancy)
{
  Kokkos::parallel_for(str, policy, functor);
}

template <class FunctorType, class ExecPolicy>

inline std::enable_if_t<
    std::is_same<typename Kokkos::Impl::FunctorPolicyExecutionSpace<
                     FunctorType, ExecPolicy>::execution_space,
                 Kokkos::Cuda>{}>
parallel_for(const std::string &str, const ExecPolicy &policy,
             const FunctorType &functor, BlockSize block_size,
             Occupancy desired_occupancy)
{
#if defined(KOKKOS_ENABLE_PROFILING)
  uint64_t kpID = 0;
  if (Kokkos::Profiling::profileLibraryLoaded())
  {
    Kokkos::Impl::ParallelConstructName<FunctorType,
                                        typename ExecPolicy::work_tag>
        name(str);
    Kokkos::Profiling::beginParallelFor(
        name.get(), Kokkos::Profiling::Experimental::device_id(policy.space()),
        &kpID);
  }
#else
  (void)str;
#endif

  Kokkos::Impl::shared_allocation_tracking_disable();
  ParallelFor<FunctorType, ExecPolicy> closure(functor, policy, block_size,
                                               desired_occupancy);
  Kokkos::Impl::shared_allocation_tracking_enable();

  closure.execute();

#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded())
  {
    Kokkos::Profiling::endParallelFor(kpID);
  }
#endif
}

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>>
{
public:
  typedef Kokkos::RangePolicy<Traits...> Policy;

private:
  typedef typename Policy::member_type Member;
  typedef typename Policy::work_tag WorkTag;
  typedef typename Policy::launch_bounds LaunchBounds;

  FunctorType const m_functor;
  Policy const m_policy;
  BlockSize const m_block_size;
  Occupancy const m_occ;

  ParallelFor() = delete;
  ParallelFor &operator=(const ParallelFor &) = delete;

  template <class TagType>
  __device__ std::enable_if_t<std::is_same<TagType, void>{}>
  exec_range(const Member i) const
  {
    m_functor(i);
  }

  template <class TagType>
  __device__ std::enable_if_t<!std::is_same<TagType, void>{}>
  exec_range(const Member i) const
  {
    m_functor(TagType(), i);
  }

public:
  typedef FunctorType functor_type;

  inline __device__ void operator()(void) const
  {
    const Member work_stride = blockDim.y * gridDim.x;
    const Member work_end = m_policy.end();

    for (Member iwork =
             m_policy.begin() + threadIdx.y + blockDim.y * blockIdx.x;
         iwork < work_end;
         iwork = iwork < work_end - work_stride ? iwork + work_stride
                                                : work_end)
    {
      this->template exec_range<WorkTag>(iwork);
    }
  }

  void execute() const
  {
    const typename Policy::index_type nwork = m_policy.end() - m_policy.begin();

    dim3 block(1, m_block_size, 1);
    dim3 grid(
        std::min(typename Policy::index_type((nwork + block.y - 1) / block.y),
                 typename Policy::index_type(
                     Kokkos::Impl::cuda_internal_maximum_grid_count())),
        1, 1);

    cudaDeviceProp const &device_prop = Kokkos::Cuda().cuda_device_prop();
    cudaFuncAttributes func_attributes;
    cudaFuncGetAttributes(&func_attributes,
                          Kokkos::Impl::cuda_parallel_launch_constant_memory<
                              ParallelFor<FunctorType, Policy>>);
    size_t const shmem =
        KokkosExt::Impl::cuda_desired_occupancy_to_dynamic_shmem(
            device_prop, func_attributes, m_block_size,
            m_occ * device_prop.maxThreadsPerMultiProcessor / 100);

    printf("occ %d shmem %zu\n",
           m_occ * device_prop.maxThreadsPerMultiProcessor / 100, shmem);
    Kokkos::Impl::CudaParallelLaunch<ParallelFor, LaunchBounds>(
        *this, grid, block, shmem,
        m_policy.space().impl_internal_space_instance(), false);
  }

  ParallelFor(FunctorType const &functor, Policy const &policy,
              BlockSize block_size, Occupancy occ)
      : m_functor(functor)
      , m_policy(policy)
      , m_block_size{block_size}
      , m_occ{occ}
  {
  }
};

} // namespace DoNotTryThisAtHome

} // namespace KokkosExt
#endif // DOXYGEN_SHOULD_SKIP_THIS

#endif
