# namespace ArborX



 Conveniently importing Point and Box in ArborX::Details:: namespace and declaring type aliases within boost::geometry:: so that we are able to just use details:: and bg:: to specify what geometry or algorithm we mean.

 We need a forward-declaration for NVCC, see below.



## Namespaces

* [Details](Details/index.md)
* [Traits](Traits/index.md)
* [Experimental](Experimental/index.md)
* [DBSCAN](DBSCAN/index.md)


## Records

* [Point](Point.md)
* [Box](Box.md)
* [Sphere](Sphere.md)
* [Nearest](Nearest.md)
* [Intersects](Intersects.md)
* [PredicateWithAttachment](PredicateWithAttachment.md)
* [PrimitivesTag](PrimitivesTag.md)
* [PredicatesTag](PredicatesTag.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [SearchException](SearchException.md)
* [AccessTraits](AccessTraits.md)
* [BasicBoundingVolumeHierarchy](BasicBoundingVolumeHierarchy.md)
* [BasicBoundingVolumeHierarchy](BasicBoundingVolumeHierarchy.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [BruteForce](BruteForce.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)
* [AccessTraits](AccessTraits.md)


## Functions

### nearest

*Nearest<Geometry> nearest(const Geometry & geometry, int k)*

*Defined at src/details/ArborX_Predicates.hpp#70*

### intersects

*Intersects<Geometry> intersects(const Geometry & geometry)*

*Defined at src/details/ArborX_Predicates.hpp#77*

### getK

*int getK(const Nearest<Geometry> & pred)*

*Defined at src/details/ArborX_Predicates.hpp#83*

### getGeometry

*const Geometry & getGeometry(const Nearest<Geometry> & pred)*

*Defined at src/details/ArborX_Predicates.hpp#89*

### getGeometry

*const Geometry & getGeometry(const Intersects<Geometry> & pred)*

*Defined at src/details/ArborX_Predicates.hpp#96*

### getData

*const Data & getData(const PredicateWithAttachment<Predicate, Data> & pred)*

*Defined at src/details/ArborX_Predicates.hpp#121*

### getPredicate

*const Predicate & getPredicate(const PredicateWithAttachment<Predicate, Data> & pred)*

*Defined at src/details/ArborX_Predicates.hpp#128*

### attach

*auto attach(Predicate && pred, Data && data)*

*Defined at src/details/ArborX_Predicates.hpp#135*

### exclusivePrefixSum

*void exclusivePrefixSum(ExecutionSpace && space, const Kokkos::View<ST, SP...> & src, const Kokkos::View<DT, DP...> & dst)*

*Defined at src/details/ArborX_DetailsUtils.hpp#186*



**brief** Computes an exclusive scan.



**space** [in]

**src** [in]

**dst** [out]

  When **p**  is not provided or if **p**  and **p**  are the same view, the  scan is performed in-place.  "Exclusive" means that the i-th input element  is not included in the i-th sum.



**pre****p**  and **p**  must be of rank 1 and have the same size.

### exclusivePrefixSum

*std::enable_if_t<Kokkos::is_execution_space<std::remove_reference_t<ExecutionSpace> >::value> exclusivePrefixSum(ExecutionSpace && space, const Kokkos::View<T, P...> & v)*

*Defined at src/details/ArborX_DetailsUtils.hpp#222*



**brief** In-place exclusive scan.



**space** [in]

**v** [in,out]

  Calls **c**  v)

### exclusivePrefixSum

*void exclusivePrefixSum(const Kokkos::View<ST, SP...> & src, const Kokkos::View<DT, DP...> & dst)*

*Defined at src/details/ArborX_DetailsUtils.hpp#230*

### exclusivePrefixSum

*void exclusivePrefixSum(const Kokkos::View<T, P...> & v)*

*Defined at src/details/ArborX_DetailsUtils.hpp#239*

### lastElement

*typename Kokkos::ViewTraits<T, P...>::non_const_value_type lastElement(const Kokkos::View<T, P...> & v)*

*Defined at src/details/ArborX_DetailsUtils.hpp#254*



**brief** Get a copy of the last element.

  Returns a copy of the last element in the view on the host.  Note that it  may require communication between host and device (e.g. if the view passed  as an argument lives on the device).



**pre****c**  is of rank 1 and not empty.

### iota

*void iota(ExecutionSpace && space, const Kokkos::View<T, P...> & v, typename Kokkos::ViewTraits<T, P...>::value_type value)*

*Defined at src/details/ArborX_DetailsUtils.hpp#279*



**brief** Fills the view with a sequence of numbers



**space** [in]

**v** [out]

**value** [in]



**note** Similar to **c**  but differs in that it directly assigns<code>

v(i) = value + i</code>

 instead of repetitively evaluating<code>

++value</code>

 which would be difficult to achieve in a performant  manner while still guaranteeing the order of execution.

### iota

*void iota(const Kokkos::View<T, P...> & v, typename Kokkos::ViewTraits<T, P...>::value_type value)*

*Defined at src/details/ArborX_DetailsUtils.hpp#299*

### minMax

*std::pair<typename ViewType::non_const_value_type, typename ViewType::non_const_value_type> minMax(ExecutionSpace && space, const ViewType & v)*

*Defined at src/details/ArborX_DetailsUtils.hpp#316*



**brief** Returns the smallest and the greatest element in the view



**space** [in]

**v** [in]

  Returns a pair on the host with the smallest value in the view as the first  element and the greatest as the second.

### minMax

*std::pair<typename ViewType::non_const_value_type, typename ViewType::non_const_value_type> minMax(const ViewType & v)*

*Defined at src/details/ArborX_DetailsUtils.hpp#346*

### min

*typename ViewType::non_const_value_type min(ExecutionSpace && space, const ViewType & v)*

*Defined at src/details/ArborX_DetailsUtils.hpp#360*



**brief** Returns the smallest element in the view



**space** [in]

**v** [in]

### min

*typename ViewType::non_const_value_type min(const ViewType & v)*

*Defined at src/details/ArborX_DetailsUtils.hpp#380*

### max

*typename ViewType::non_const_value_type max(ExecutionSpace && space, const ViewType & v)*

*Defined at src/details/ArborX_DetailsUtils.hpp#393*



**brief** Returns the greatest element in the view



**space** [in]

**v** [in]

### max

*typename ViewType::non_const_value_type max(const ViewType & v)*

*Defined at src/details/ArborX_DetailsUtils.hpp#413*

### accumulate

*typename ViewType::non_const_value_type accumulate(ExecutionSpace && space, const ViewType & v, typename ViewType::non_const_value_type init)*

*Defined at src/details/ArborX_DetailsUtils.hpp#430*



**brief** Accumulate values in a view



**space** [in]

**v** [in]

**init** [in]

  Returns the sum of the given **p**  value and elements in the given view **p**   Uses operator+ to sum up the elements.

### accumulate

*typename ViewType::non_const_value_type accumulate(const ViewType & v, typename ViewType::non_const_value_type init)*

*Defined at src/details/ArborX_DetailsUtils.hpp#456*

### clone

*typename View::non_const_type clone(View & v)*

*Defined at src/details/ArborX_DetailsUtils.hpp#582*

 FIXME shameless forward declaration

### adjacentDifference

*void adjacentDifference(ExecutionSpace && space, const SrcViewType & src, const DstViewType & dst)*

*Defined at src/details/ArborX_DetailsUtils.hpp#480*



**brief** Computes the adjacent difference.



**space** [in]

**src** [in]

**dst** [out]

  Assigns to every element in the **p**  view the difference between its  corresponding element and the one preceding it in the **p**  view, except  for the first element **c**  which is assigned **c** 



**warning** Undefined behavior if **p**  and **p**  arrays overlap in any way.

### adjacentDifference

*void adjacentDifference(const SrcViewType & src, const DstViewType & dst)*

*Defined at src/details/ArborX_DetailsUtils.hpp#508*

### reallocWithoutInitializing

*void reallocWithoutInitializing(View & v, size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7)*

*Defined at src/details/ArborX_DetailsUtils.hpp#531*

 NOTE: not possible to avoid initialization with Kokkos::realloc()

### reallocWithoutInitializing

*void reallocWithoutInitializing(View & v, const typename View::array_layout & layout)*

*Defined at src/details/ArborX_DetailsUtils.hpp#558*

### cloneWithoutInitializingNorCopying

*typename View::non_const_type cloneWithoutInitializingNorCopying(View & v)*

*Defined at src/details/ArborX_DetailsUtils.hpp#566*

### clone

*typename View::non_const_type clone(ExecutionSpace && space, View & v)*

*Defined at src/details/ArborX_DetailsUtils.hpp#573*

### query

*void query(const Tree & tree, const ExecutionSpace & space, const Predicates & predicates, CallbackOrView && callback_or_view, View && view, Args &&... args)*

*Defined at src/ArborX_CrsGraphWrapper.hpp#22*

### query

*void query(const Tree & tree, const ExecutionSpace & space, const Predicates & predicates, CallbackOrView && callback_or_view, View && view, Args &&... args)*

### dbscan

*Kokkos::View<int *, typename AccessTraits<Primitives, PrimitivesTag>::memory_space> dbscan(const ExecutionSpace & exec_space, const Primitives & primitives, float eps, int core_min_size, const DBSCAN::Parameters & parameters)*

*Defined at src/ArborX_DBSCAN.hpp#195*

### version

*basic_string version()*

*Defined at build/include/ArborX_Version.hpp#24*

### gitCommitHash

*basic_string gitCommitHash()*

*Defined at build/include/ArborX_Version.hpp#26*

### query

*void query(const BoostExt::RTree<Indexable> & rtree, const ExecutionSpace & space, const Predicates & predicates, InputView & indices, InputView & offset, TrailingArgs &&... args)*

*Defined at test/ArborX_BoostRTreeHelpers.hpp#327*

 Specialization of ArborX::query



## Enums

| enum class CallbackTreeTraversalControl |

--

| early_exit |
| normal_continuation |


*Defined at src/details/ArborX_Callbacks.hpp#24*



