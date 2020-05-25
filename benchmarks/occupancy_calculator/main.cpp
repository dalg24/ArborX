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

#include <ArborX_CudaOccupancy.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <benchmark/benchmark.h>

struct Foo
{
  KOKKOS_FUNCTION void operator()(int) const {}
  size_t team_shmem_size(int team_size) const { return 128 * team_size; }
  ArborX::BVH<Kokkos::CudaSpace> bvh;
  Kokkos::View<int *, Kokkos::Cuda> predicates;
};

template <typename FunctorType>
void BM_kokkos_cuda_get_opt_block_size(benchmark::State &state)
{
  Kokkos::Impl::CudaInternal *cuda_instance =
      Kokkos::Cuda().impl_internal_space_instance();

  cudaFuncAttributes attr = Kokkos::Impl::CudaParallelLaunch<
      Kokkos::Impl::ParallelFor<FunctorType, Kokkos::RangePolicy<Kokkos::Cuda>>,
      Kokkos::LaunchBounds<>>::get_cuda_func_attributes();

  for (auto _ : state)
  {
    int const vector_length = 1;
    int const shmem_block = 0;
    int const shmem_thread = 0;

    int const block_size =
        Kokkos::Impl::cuda_get_opt_block_size<FunctorType,
                                              Kokkos::LaunchBounds<>>(
            cuda_instance, attr, FunctorType{}, vector_length, shmem_block,
            shmem_thread);

#ifdef PRINT_DEBUG
    printf("Kokkos %d\n", block_size);
    break;
#endif
    benchmark::DoNotOptimize(block_size);
  }
}

BENCHMARK_TEMPLATE(BM_kokkos_cuda_get_opt_block_size, Foo);

template <typename FunctorType>
void BM_cuda_runtime_api(benchmark::State &bm_state)
{
  cudaOccDeviceProp const properties = Kokkos::Cuda().cuda_device_prop();

  cudaOccFuncAttributes const attributes = Kokkos::Impl::CudaParallelLaunch<
      Kokkos::Impl::ParallelFor<FunctorType, Kokkos::RangePolicy<Kokkos::Cuda>>,
      Kokkos::LaunchBounds<>>::get_cuda_func_attributes();

  cudaOccDeviceState const state{};

  auto const blockSizeToDynamicSMemSize = [attributes](int block_size) {
    int const vector_length = 1;
    int const shmem_block = 0;
    int const shmem_thread = 0;

    int const functor_shmem =
        Kokkos::Impl::FunctorTeamShmemSize<FunctorType>::value(
            FunctorType{}, block_size / vector_length);

    int const total_shmem = shmem_block +
                            shmem_thread * (block_size / vector_length) +
                            functor_shmem + attributes.sharedSizeBytes;
    return total_shmem;
  };

  for (auto _ : bm_state)
  {
    int minGridSize;
    int blockSize;

    cudaOccMaxPotentialOccupancyBlockSizeVariableSMem(
        &minGridSize, &blockSize, &properties, &attributes, &state,
        blockSizeToDynamicSMemSize);

#ifdef PRINT_DEBUG
    printf("CUDA runtime %d %d\n", minGridSize, blockSize);
    break;
#endif
    benchmark::DoNotOptimize(blockSize);
  }
}

BENCHMARK_TEMPLATE(BM_cuda_runtime_api, Foo);

template <typename FunctorType>
void BM_kokkos_ext(benchmark::State &state)
{
  cudaDeviceProp const properties = Kokkos::Cuda().cuda_device_prop();

  cudaFuncAttributes const attributes = Kokkos::Impl::CudaParallelLaunch<
      Kokkos::Impl::ParallelFor<FunctorType, Kokkos::RangePolicy<Kokkos::Cuda>>,
      Kokkos::LaunchBounds<>>::get_cuda_func_attributes();

  auto const blockSizeToDynamicSMemSize = [attributes](int block_size) {
    int const vector_length = 1;
    int const shmem_block = 0;
    int const shmem_thread = 0;

    int const functor_shmem =
        Kokkos::Impl::FunctorTeamShmemSize<FunctorType>::value(
            FunctorType{}, block_size / vector_length);

    int const total_shmem = shmem_block +
                            shmem_thread * (block_size / vector_length) +
                            functor_shmem + attributes.sharedSizeBytes;
    return total_shmem;
  };

  for (auto _ : state)
  {
    int blockSize = KokkosExt::Impl::cuda_min_block_size_maximize_occupancy(
        properties, attributes, blockSizeToDynamicSMemSize, 0, 32);

#ifdef PRINT_DEBUG
    printf("KokkosExt %d\n", blockSize);
    break;
#endif
    benchmark::DoNotOptimize(blockSize);
  }
}

BENCHMARK_TEMPLATE(BM_kokkos_ext, Foo);

template <typename FunctorType>
void BM_kokkos_ext_desired_occupancy(benchmark::State &bm_state)
{
  cudaDeviceProp const properties = Kokkos::Cuda().cuda_device_prop();

  cudaFuncAttributes const attributes = Kokkos::Impl::CudaParallelLaunch<
      Kokkos::Impl::ParallelFor<FunctorType, Kokkos::RangePolicy<Kokkos::Cuda>>,
      Kokkos::LaunchBounds<>>::get_cuda_func_attributes();

  cudaOccDeviceState const state{};

  auto const blockSizeToDynamicSMemSize = [attributes](int block_size) {
    int const vector_length = 1;
    int const shmem_block = 0;
    int const shmem_thread = 0;

    int const functor_shmem =
        Kokkos::Impl::FunctorTeamShmemSize<FunctorType>::value(
            FunctorType{}, block_size / vector_length);

    int const total_shmem = shmem_block +
                            shmem_thread * (block_size / vector_length) +
                            functor_shmem + attributes.sharedSizeBytes;
    return total_shmem;
  };

  for (auto _ : bm_state)
  {
    size_t sharedMemPerBlock =
        KokkosExt::Impl::cuda_desired_occupancy_to_dynamic_shmem(
            properties, attributes, 128, 512);

#ifdef PRINT_DEBUG
    printf("Desired occupancy %d\n", sharedMemPerBlock);
    break;
#endif
    benchmark::DoNotOptimize(sharedMemPerBlock);
  }
}

BENCHMARK_TEMPLATE(BM_kokkos_ext_desired_occupancy, Foo);

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  benchmark::Initialize(&argc, argv);

  benchmark::RunSpecifiedBenchmarks();

  return 0;
}
