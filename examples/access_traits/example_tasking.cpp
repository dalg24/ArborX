#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <iostream>
#include <numeric>

struct PointCloud
{
  float *d_x;
  float *d_y;
  float *d_z;
  int N;
};

struct Spheres
{
  float *d_x;
  float *d_y;
  float *d_z;
  float *d_r;
  int N;
};

namespace ArborX
{
namespace Traits
{
template <>
struct Access<PointCloud, PrimitivesTag>
{
  using memory_space = Kokkos::HostSpace;
  inline static std::size_t size(PointCloud const &cloud) { return cloud.N; }
  KOKKOS_FUNCTION static Point get(PointCloud const &cloud, std::size_t i)
  {
    return {{cloud.d_x[i], cloud.d_y[i], cloud.d_z[i]}};
  }
};

template <>
struct Access<Spheres, PredicatesTag>
{
  using memory_space = Kokkos::HostSpace;
  inline static std::size_t size(Spheres const &d) { return d.N; }
  KOKKOS_FUNCTION static auto get(Spheres const &d, std::size_t i)
  {
    return intersects(Sphere{{{d.d_x[i], d.d_y[i], d.d_z[i]}}, d.d_r[i]});
  }
};
} // namespace Traits
} // namespace ArborX

int main(int argc, char *argv[])
{
  Kokkos::initialize(argc, argv);
  {

    constexpr std::size_t N = 10;
    std::array<float, N> a;

    float *d_a = a.data();
    std::iota(std::begin(a), std::end(a), 1.0);

    using device_type = Kokkos::Serial::device_type;
    ArborX::BVH<device_type> bvh{PointCloud{d_a, d_a, d_a, N}};

    ArborX::Details::TreeTraversal<device_type>::spatialQueryExperimental(
        bvh, ArborX::intersects(ArborX::Box{{{0, 0, 0}}, {{10, 10, 10}}}),
        KOKKOS_LAMBDA(int i) { printf("found %d\n", i); });

    std::cout << "BATCHED VERSION TASKING\n";
    Spheres dummy{d_a, d_a, d_a, d_a, N};
    using Access =
        ArborX::Traits::Access<Spheres, ArborX::Traits::PredicatesTag>;
    int const n_predicates = Access::size(dummy);
    Kokkos::View<ArborX::Intersects<ArborX::Sphere> *, Kokkos::HostSpace>
        predicates(Kokkos::ViewAllocateWithoutInitializing("predicates"),
                   n_predicates);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::OpenMP>(0, n_predicates),
        KOKKOS_LAMBDA(int i) { predicates(i) = Access::get(dummy, i); });

    Kokkos::View<int *, device_type> indices(
        Kokkos::view_alloc("indices", Kokkos::WithoutInitializing), 0);
    Kokkos::View<int *, device_type> offset(
        Kokkos::view_alloc("offset", Kokkos::WithoutInitializing), 0);
    ArborX::Details::TreeTraversal<
        device_type>::batchedSpatialQueryExperimental(bvh, predicates, indices,
                                                      offset);
    std::cout << "DONE\n";

    bvh.query(Spheres{d_a, d_a, d_a, d_a, N}, indices, offset);

    Kokkos::parallel_for(N, KOKKOS_LAMBDA(int i) {
      for (int j = offset(i); j < offset(i + 1); ++j)
      {
        printf("%i %i\n", i, indices(j));
      }
    });
  }
  Kokkos::finalize();

  return 0;
}
