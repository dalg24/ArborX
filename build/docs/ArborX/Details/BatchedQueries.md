# struct BatchedQueries

*Defined at src/details/ArborX_DetailsBatchedQueries.hpp#32*

## Functions

### sortQueriesAlongZOrderCurve

*public Kokkos::View<unsigned int *, DeviceType> sortQueriesAlongZOrderCurve(const ExecutionSpace & space, const struct ArborX::Box & scene_bounding_box, const Predicates & predicates)*

*Defined at src/details/ArborX_DetailsBatchedQueries.hpp#50*

 BatchedQueries defines functions for sorting queries along the Z-order space-filling curve in order to minimize data divergence.  The goal is to increase correlation between traversal decisions made by nearby threads and thereby increase performance.

 NOTE: sortQueriesAlongZOrderCurve() does not actually apply the sorting order, it returns the permutation indices.  applyPermutation() was added in that purpose.  reversePermutation() is able to restore the initial order on the results that are in "compressed row storage" format.  You may notice it is not used any more in the code that performs the batched queries.  We found that it was slighly more performant to add a level of indirection when recording results rather than using that function at the end.  We decided to keep reversePermutation around for now.

### applyPermutation

*public Kokkos::View<typename AccessTraitsHelper<AccessTraits<Predicates, PredicatesTag> >::type *, DeviceType> applyPermutation(const ExecutionSpace & space, Kokkos::View<const unsigned int *, DeviceType> permute, const Predicates & v)*

*Defined at src/details/ArborX_DetailsBatchedQueries.hpp#79*

 NOTE  trailing return type seems required :( error: The enclosing parent function ("applyPermutation") for an extended __host__ __device__ lambda must not have deduced return type

### permuteOffset

*public typename Offset::non_const_type permuteOffset(const ExecutionSpace & space, const Permute & permute, const Offset & offset)*

*Defined at src/details/ArborX_DetailsBatchedQueries.hpp#104*

### permuteIndices

*public typename Values::non_const_type permuteIndices(const ExecutionSpace & space, const Permute & permute, const Values & indices, const Offset & offset, const Offset2 & tmp_offset)*

*Defined at src/details/ArborX_DetailsBatchedQueries.hpp#125*

### reversePermutation

*public std::tuple<Kokkos::View<int *, DeviceType>, Kokkos::View<T *, P...> > reversePermutation(const ExecutionSpace & space, Kokkos::View<const unsigned int *, DeviceType> permute, Kokkos::View<const int *, DeviceType> offset, Kokkos::View<T *, P...> out)*

*Defined at src/details/ArborX_DetailsBatchedQueries.hpp#150*

### reversePermutation

*public std::tuple<Kokkos::View<int *, DeviceType>, Kokkos::View<int *, DeviceType>, Kokkos::View<float *, DeviceType> > reversePermutation(const ExecutionSpace & space, Kokkos::View<const unsigned int *, DeviceType> permute, Kokkos::View<const int *, DeviceType> offset, Kokkos::View<const int *, DeviceType> indices, Kokkos::View<const float *, DeviceType> distances)*

*Defined at src/details/ArborX_DetailsBatchedQueries.hpp#164*



