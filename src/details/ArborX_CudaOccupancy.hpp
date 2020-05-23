#ifndef ARBORX_CUDA_OCCUPANCY_HPP
#define ARBORX_CUDA_OCCUPANCY_HPP

#include <vector>

#include <cuda_occupancy.h>

namespace KokkosExt
{
namespace Impl
{

static inline int
cuda_max_active_blocks_per_sm(cudaDeviceProp const &properties,
                              cudaFuncAttributes const &attributes,
                              int block_size, size_t dynamic_shmem)
{
  // Limits due do registers/SM
  int const regs_per_sm = properties.regsPerMultiprocessor;
  int const regs_per_thread = attributes.numRegs;
  int const max_blocks_regs = regs_per_sm / (regs_per_thread * block_size);
  // printf("regs/sm %d  regs/thread %d  occ %d\n", regs_per_sm,
  // regs_per_thread,
  //       max_blocks_regs);

  // Limits due to shared memory/SM
  size_t const shmem_per_sm = properties.sharedMemPerMultiprocessor;
  size_t const shmem_per_block = properties.sharedMemPerBlock;
  size_t const static_shmem = attributes.sharedSizeBytes;
  size_t const dynamic_shmem_per_block = attributes.maxDynamicSharedSizeBytes;
  size_t const total_shmem = static_shmem + dynamic_shmem;

  int const max_blocks_shmem =
      total_shmem > shmem_per_block || dynamic_shmem > dynamic_shmem_per_block
          ? 0
          : (total_shmem > 0 ? (int)shmem_per_sm / total_shmem
                             : max_blocks_regs);
  // printf("shmem/sm %d  shmem/cta %d  staticshmem %d  dynamicshmem/block %d  "
  //       "occ %d\n",
  //       shmem_per_sm, shmem_per_block, static_shmem, dynamic_shmem_per_block,
  //       max_blocks_shmem);

  // Overall occupancy in blocks
  return std::min(max_blocks_regs, max_blocks_shmem);
}

template <typename UnaryFunction>
inline int cuda_min_block_size_maximize_occupancy(
    cudaDeviceProp const &properties, cudaFuncAttributes const &attributes,
    UnaryFunction block_size_to_dynamic_shmem, int min_blocks_per_sm,
    int max_blocks_per_sm)
{
  // Limits
  int const max_threads_per_sm = properties.maxThreadsPerMultiProcessor;
  int const max_threads_per_block =
      std::min(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);

  // Recorded maximum
  int opt_block_size = 0;
  int opt_threads_per_sm = 0;

  for (int block_size = max_threads_per_sm; block_size > 0; block_size -= 32)
  {
    size_t const dynamic_shmem = block_size_to_dynamic_shmem(block_size);

    int blocks_per_sm = cuda_max_active_blocks_per_sm(
        properties, attributes, block_size, dynamic_shmem);

    int threads_per_sm = blocks_per_sm * block_size;

    if (threads_per_sm > max_threads_per_sm)
    {
      threads_per_sm = max_threads_per_sm;
      blocks_per_sm = threads_per_sm / block_size;
    }

    if (blocks_per_sm >= min_blocks_per_sm &&
        blocks_per_sm <= max_blocks_per_sm)
    {
      if (threads_per_sm >= opt_threads_per_sm)
      {
        opt_block_size = block_size;
        opt_threads_per_sm = threads_per_sm;
      }
    }
  }

  return opt_block_size;
}

struct Up
{
};

static inline std::vector<size_t> supportedSharedMemPerMultiprocessorCapacities(
    cudaOccDeviceProp const &properties)
{
  switch (properties.computeMajor)
  {
  case 7:
  {
    // Turing supports 32KB and 64KB shared mem.
    int isTuring = properties.computeMinor == 5;
    if (isTuring)
    {
      return {32768, properties.sharedMemPerMultiprocessor};
    }
    // Volta supports 0KB, 8KB, 16KB, 32KB, 64KB, and 96KB shared mem.
    else
    {
      return {0, 8192, 16384, 32768, properties.sharedMemPerMultiprocessor};
    }
    break;
  }
  default:
    throw Up{};
  }
}

static inline size_t
cuda_desired_occupancy_to_dynamic_shmem(cudaDeviceProp const &properties,
                                        cudaFuncAttributes const &attributes,
                                        int block_size, int desired_occupancy)
{
  int opt_delta = properties.maxThreadsPerMultiProcessor;

  size_t opt_dynamic_shmem = 0;

  for (size_t dynamic_shmem :
       supportedSharedMemPerMultiprocessorCapacities(properties))
  {
    int blocks_per_sm = cuda_max_active_blocks_per_sm(
        properties, attributes, block_size, dynamic_shmem);

    int threads_per_sm = blocks_per_sm * block_size;

    int delta = std::abs(threads_per_sm - desired_occupancy);

    if (delta < opt_delta)
    {
      opt_delta = delta;
      opt_dynamic_shmem = dynamic_shmem;
    }
  }

  return opt_dynamic_shmem;
}

} // namespace Impl
} // namespace KokkosExt

#endif
