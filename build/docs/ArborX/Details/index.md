# namespace Details



## Namespaces

* [internal](internal/index.md)
* [PermuteHelper](PermuteHelper/index.md)
* [CrsGraphWrapperImpl](CrsGraphWrapperImpl/index.md)
* [TreeConstruction](TreeConstruction/index.md)


## Records

* [NearestPredicateTag](NearestPredicateTag.md)
* [SpatialPredicateTag](SpatialPredicateTag.md)
* [AccessTraitsHelper](AccessTraitsHelper.md)
* [AccessTraitsHelper](AccessTraitsHelper.md)
* [InlineCallbackTag](InlineCallbackTag.md)
* [PostCallbackTag](PostCallbackTag.md)
* [DefaultCallback](DefaultCallback.md)
* [is_tagged_post_callback](is_tagged_post_callback.md)
* [Sink](Sink.md)
* [invoke_callback_and_check_early_exit_helper](invoke_callback_and_check_early_exit_helper.md)
* [ExclusiveScanFunctor](ExclusiveScanFunctor.md)
* [BatchedQueries](BatchedQueries.md)
* [PermutedData](PermutedData.md)
* [FirstPassTag](FirstPassTag.md)
* [FirstPassNoBufferOptimizationTag](FirstPassNoBufferOptimizationTag.md)
* [SecondPassTag](SecondPassTag.md)
* [InsertGenerator](InsertGenerator.md)
* [NodeWithTwoChildrenTag](NodeWithTwoChildrenTag.md)
* [NodeWithLeftChildAndRopeTag](NodeWithLeftChildAndRopeTag.md)
* [NodeWithTwoChildren](NodeWithTwoChildren.md)
* [NodeWithLeftChildAndRope](NodeWithLeftChildAndRope.md)
* [HappyTreeFriends](HappyTreeFriends.md)
* [StaticVector](StaticVector.md)
* [UnmanagedStaticVector](UnmanagedStaticVector.md)
* [Less](Less.md)
* [Greater](Greater.md)
* [PriorityQueue](PriorityQueue.md)
* [Stack](Stack.md)
* [TreeTraversal](TreeTraversal.md)
* [TreeTraversal](TreeTraversal.md)
* [TreeTraversal](TreeTraversal.md)
* [BruteForceImpl](BruteForceImpl.md)
* [UnionFind](UnionFind.md)
* [CountUpToN](CountUpToN.md)
* [FDBSCANCallback](FDBSCANCallback.md)
* [CartesianGrid](CartesianGrid.md)
* [CountUpToN_DenseBox](CountUpToN_DenseBox.md)
* [FDBSCANDenseBoxCallback](FDBSCANDenseBoxCallback.md)
* [CCSCorePoints](CCSCorePoints.md)
* [DBSCANCorePoints](DBSCANCorePoints.md)
* [PrimitivesWithRadius](PrimitivesWithRadius.md)
* [PrimitivesWithRadiusReorderedAndFiltered](PrimitivesWithRadiusReorderedAndFiltered.md)
* [MixedBoxPrimitives](MixedBoxPrimitives.md)
* [TreeVisualization](TreeVisualization.md)
* [Direction](Direction.md)
* [KDOP_Directions](KDOP_Directions.md)
* [KDOP_Directions](KDOP_Directions.md)
* [KDOP_Directions](KDOP_Directions.md)
* [KDOP_Directions](KDOP_Directions.md)
* [KDOP_Directions](KDOP_Directions.md)


## Functions

### equals

*_Bool equals(const class ArborX::Point & l, const class ArborX::Point & r)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#27*

### equals

*_Bool equals(const struct ArborX::Box & l, const struct ArborX::Box & r)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#36*

### equals

*_Bool equals(const struct ArborX::Sphere & l, const struct ArborX::Sphere & r)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#43*

### isValid

*_Bool isValid(const class ArborX::Point & p)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#49*

### isValid

*_Bool isValid(const struct ArborX::Box & b)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#59*

### isValid

*_Bool isValid(const struct ArborX::Sphere & s)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#72*

### distance

*float distance(const class ArborX::Point & a, const class ArborX::Point & b)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#80*

 distance point-point

### distance

*float distance(const class ArborX::Point & point, const struct ArborX::Box & box)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#93*

 distance point-box

### distance

*float distance(const class ArborX::Point & point, const struct ArborX::Sphere & sphere)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#110*

 distance point-sphere

### distance

*float distance(const struct ArborX::Box & box_a, const struct ArborX::Box & box_b)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#118*

 distance box-box

### distance

*float distance(const struct ArborX::Sphere & sphere, const struct ArborX::Box & box)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#147*

 distance box-sphere

### expand

*void expand(struct ArborX::Box & box, const class ArborX::Point & point)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#157*

 expand an axis-aligned bounding box to include a point

### expand

*void expand(BOX & box, const BOX & other)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#167*

 expand an axis-aligned bounding box to include another box NOTE: Box type is templated here to be able to use expand(box, box) in a Kokkos::parallel_reduce() in which case the arguments must be declared volatile.

### expand

*void expand(struct ArborX::Box & box, const struct ArborX::Sphere & sphere)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#173*

 expand an axis-aligned bounding box to include a sphere

### intersects

*_Bool intersects(const struct ArborX::Box & box, const struct ArborX::Box & other)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#188*

 check if two axis-aligned bounding boxes intersect

### intersects

*_Bool intersects(const class ArborX::Point & point, const struct ArborX::Box & other)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#198*

### intersects

*_Bool intersects(const struct ArborX::Sphere & sphere, const struct ArborX::Box & box)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#208*

 check if a sphere intersects with an  axis-aligned bounding box

### intersects

*_Bool intersects(const struct ArborX::Sphere & sphere, const class ArborX::Point & point)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#214*

### intersects

*_Bool intersects(const class ArborX::Point & point, const struct ArborX::Sphere & sphere)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#220*

### centroid

*void centroid(const struct ArborX::Box & box, class ArborX::Point & c)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#227*

 calculate the centroid of a box

### centroid

*void centroid(const class ArborX::Point & point, class ArborX::Point & c)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#234*

### centroid

*void centroid(const struct ArborX::Sphere & sphere, class ArborX::Point & c)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#237*

### returnCentroid

*Point returnCentroid(const class ArborX::Point & point)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#240*

### returnCentroid

*Point returnCentroid(const struct ArborX::Box & box)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#243*

### returnCentroid

*Point returnCentroid(const struct ArborX::Sphere & sphere)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#252*

### translateAndScale

*void translateAndScale(const class ArborX::Point & in, class ArborX::Point & out, const struct ArborX::Box & ref)*

*Defined at src/details/ArborX_DetailsAlgorithms.hpp#257*

 transformation that maps the unit cube into a new axis-aligned box NOTE safe to perform in-place

### check_valid_access_traits

*void check_valid_access_traits(PredicatesTag , const Predicates & )*

*Defined at src/details/ArborX_AccessTraits.hpp#110*

### check_valid_access_traits

*void check_valid_access_traits(PrimitivesTag , const Primitives & )*

*Defined at src/details/ArborX_AccessTraits.hpp#150*

### check_generic_lambda_support

*void check_generic_lambda_support(const Callback & )*

*Defined at src/details/ArborX_Callbacks.hpp#86*

### check_valid_callback

*void check_valid_callback(const Callback & callback, const Predicates & , const OutputView & )*

*Defined at src/details/ArborX_Callbacks.hpp#97*

### invoke_callback_and_check_early_exit

*std::enable_if_t<invoke_callback_and_check_early_exit_helper<std::decay_t<Callback>, std::decay_t<Predicate>, std::decay_t<Primitive> >::value, _Bool> invoke_callback_and_check_early_exit(Callback && callback, Predicate && predicate, Primitive && primitive)*

*Defined at src/details/ArborX_Callbacks.hpp#148*

 Invoke a callback that may return a hint to interrupt the tree traversal and return true for early exit, or false for normal continuation.

### invoke_callback_and_check_early_exit

*std::enable_if_t<!invoke_callback_and_check_early_exit_helper<std::decay_t<Callback>, std::decay_t<Predicate>, std::decay_t<Primitive> >::value, _Bool> invoke_callback_and_check_early_exit(Callback && callback, Predicate && predicate, Primitive && primitive)*

*Defined at src/details/ArborX_Callbacks.hpp#165*

 Invoke a callback that does not return a hint.  Always return false to signify that the tree traversal should continue normally.

### check_valid_callback

*void check_valid_callback(const Callback & callback, const Predicates & )*

*Defined at src/details/ArborX_Callbacks.hpp#179*

### expandBits

*unsigned int expandBits(unsigned int v)*

*Defined at src/details/ArborX_DetailsMortonCode.hpp#26*

 Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.

### morton3D

*unsigned int morton3D(double x, double y, double z)*

*Defined at src/details/ArborX_DetailsMortonCode.hpp#38*

 Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].

### create_layout_right_mirror_view

*Kokkos::View<typename View::traits::data_type, Kokkos::LayoutRight, typename ExecutionSpace::memory_space> create_layout_right_mirror_view(const ExecutionSpace & , const View & src, typename std::enable_if<!((std::is_same<typename View::traits::array_layout, Kokkos::LayoutRight>::value || (View::rank == 1 && !std::is_same<typename View::traits::array_layout, Kokkos::LayoutStride>::value)) && std::is_same<typename View::traits::memory_space, typename ExecutionSpace::memory_space>::value)>::type * )*

*Defined at src/details/ArborX_DetailsUtils.hpp#47*

### create_layout_right_mirror_view

*auto create_layout_right_mirror_view(const ExecutionSpace & , const View & src, typename std::enable_if<((std::is_same<typename View::traits::array_layout, Kokkos::LayoutRight>::value || (View::rank == 1 && !std::is_same<typename View::traits::array_layout, Kokkos::LayoutStride>::value)) && std::is_same<typename View::traits::memory_space, typename ExecutionSpace::memory_space>::value)>::type * )*

*Defined at src/details/ArborX_DetailsUtils.hpp#75*

### create_layout_right_mirror_view

*auto create_layout_right_mirror_view(const View & src)*

*Defined at src/details/ArborX_DetailsUtils.hpp#90*

### create_layout_right_mirror_view_and_copy

*auto create_layout_right_mirror_view_and_copy(const ExecutionSpace & execution_space, const View & src, typename std::enable_if<!((std::is_same<typename View::traits::array_layout, Kokkos::LayoutRight>::value || (View::rank == 1 && !std::is_same<typename View::traits::array_layout, Kokkos::LayoutStride>::value)) && std::is_same<typename View::traits::memory_space, typename ExecutionSpace::memory_space>::value)>::type * )*

*Defined at src/details/ArborX_DetailsUtils.hpp#97*

### create_layout_right_mirror_view_and_copy

*auto create_layout_right_mirror_view_and_copy(const ExecutionSpace & , const View & src, typename std::enable_if<((std::is_same<typename View::traits::array_layout, Kokkos::LayoutRight>::value || (View::rank == 1 && !std::is_same<typename View::traits::array_layout, Kokkos::LayoutStride>::value)) && std::is_same<typename View::traits::memory_space, typename ExecutionSpace::memory_space>::value)>::type * )*

*Defined at src/details/ArborX_DetailsUtils.hpp#131*

### sortObjects

*Kokkos::View<SizeType *, typename ViewType::device_type> sortObjects(const ExecutionSpace & space, ViewType & view)*

*Defined at src/details/ArborX_DetailsSortUtils.hpp#78*

 NOTE returns the permutation indices **and** sorts the input view

### applyInversePermutation

*void applyInversePermutation(const ExecutionSpace & space, const PermutationView & permutation, const InputView & input_view, const OutputView & output_view)*

*Defined at src/details/ArborX_DetailsSortUtils.hpp#231*

### applyPermutation

*void applyPermutation(const ExecutionSpace & space, const PermutationView & permutation, const InputView & input_view, const OutputView & output_view)*

*Defined at src/details/ArborX_DetailsSortUtils.hpp#252*

### applyPermutation

*void applyPermutation(const ExecutionSpace & space, const PermutationView & permutation, View & view)*

*Defined at src/details/ArborX_DetailsSortUtils.hpp#272*

### toBufferStatus

*enum ArborX::Details::BufferStatus toBufferStatus(int buffer_size)*

*Defined at src/details/ArborX_DetailsCrsGraphWrapperImpl.hpp#34*

### makeLeafNode

*NodeWithTwoChildren<BoundingVolume> makeLeafNode(NodeWithTwoChildrenTag , std::size_t permutation_index, BoundingVolume bounding_volume)*

*Defined at src/details/ArborX_DetailsNode.hpp#62*

### makeLeafNode

*NodeWithLeftChildAndRope<BoundingVolume> makeLeafNode(NodeWithLeftChildAndRopeTag , std::size_t permutation_index, BoundingVolume bounding_volume)*

*Defined at src/details/ArborX_DetailsNode.hpp#109*

### isHeap

*_Bool isHeap(RandomIterator first, RandomIterator last, Compare comp)*

*Defined at src/details/ArborX_DetailsHeap.hpp#25*

### bubbleUp

*void bubbleUp(RandomIterator first, DistanceType pos, DistanceType top, ValueType val, Compare comp)*

*Defined at src/details/ArborX_DetailsHeap.hpp#42*

### pushHeap

*void pushHeap(RandomIterator first, RandomIterator last, Compare comp)*

*Defined at src/details/ArborX_DetailsHeap.hpp#57*

### bubbleDown

*void bubbleDown(RandomIterator first, DistanceType pos, DistanceType len, ValueType val, Compare comp)*

*Defined at src/details/ArborX_DetailsHeap.hpp#73*

### popHeap

*void popHeap(RandomIterator first, RandomIterator last, Compare comp)*

*Defined at src/details/ArborX_DetailsHeap.hpp#94*

### makeHeap

*void makeHeap(RandomIterator first, RandomIterator last, Compare comp)*

*Defined at src/details/ArborX_DetailsHeap.hpp#110*

### sortHeap

*void sortHeap(RandomIterator first, RandomIterator last, Compare comp)*

*Defined at src/details/ArborX_DetailsHeap.hpp#119*

### traverse

*void traverse(const ExecutionSpace & space, const BVH & bvh, const Predicates & predicates, const Callback & callback)*

*Defined at src/details/ArborX_DetailsTreeTraversal.hpp#458*

### computeCellIndices

*Kokkos::View<size_t *, typename AccessTraits<Primitives, PrimitivesTag>::memory_space> computeCellIndices(const ExecutionSpace & exec_space, const Primitives & primitives, const struct ArborX::Details::CartesianGrid & grid)*

*Defined at src/details/ArborX_DetailsFDBSCANDenseBox.hpp#240*

### computeOffsetsInOrderedView

*Kokkos::View<int *, typename View::memory_space> computeOffsetsInOrderedView(const ExecutionSpace & exec_space, View view)*

*Defined at src/details/ArborX_DetailsFDBSCANDenseBox.hpp#266*

 TODO: should put it together with other commonly used Kokkos routines and unit test it in the future

### reorderDenseAndSparseCells

*int reorderDenseAndSparseCells(const ExecutionSpace & exec_space, CellOffsets cell_offsets, int core_min_size, CellIndices & sorted_cell_indices, Permutation & permute)*

*Defined at src/details/ArborX_DetailsFDBSCANDenseBox.hpp#303*

### unionFindWithinEachDenseCell

*void unionFindWithinEachDenseCell(const ExecutionSpace & exec_space, CellIndices sorted_dense_cell_indices, Permutation permute, Labels labels)*

*Defined at src/details/ArborX_DetailsFDBSCANDenseBox.hpp#362*

### verifyCorePointsNonnegativeIndex

*_Bool verifyCorePointsNonnegativeIndex(const ExecutionSpace & exec_space, IndicesView , OffsetView offset, LabelsView labels, int core_min_size)*

*Defined at examples/dbscan/ArborX_DBSCANVerification.hpp#31*

 Check that core points have nonnegative indices

### verifyConnectedCorePointsShareIndex

*_Bool verifyConnectedCorePointsShareIndex(const ExecutionSpace & exec_space, IndicesView indices, OffsetView offset, LabelsView labels, int core_min_size)*

*Defined at examples/dbscan/ArborX_DBSCANVerification.hpp#59*

 Check that connected core points have same cluster indices

### verifyBorderAndNoisePoints

*_Bool verifyBorderAndNoisePoints(const ExecutionSpace & exec_space, IndicesView indices, OffsetView offset, LabelsView labels, int core_min_size)*

*Defined at examples/dbscan/ArborX_DBSCANVerification.hpp#99*

 Check that border points share index with at least one core point, and that noise points have index -1

### verifyClustersAreUnique

*_Bool verifyClustersAreUnique(const ExecutionSpace & exec_space, IndicesView indices, OffsetView offset, LabelsView labels, int core_min_size)*

*Defined at examples/dbscan/ArborX_DBSCANVerification.hpp#159*

 Check that cluster indices are unique

### verifyClusters

*_Bool verifyClusters(const ExecutionSpace & exec_space, IndicesView indices, OffsetView offset, LabelsView labels, int core_min_size)*

*Defined at examples/dbscan/ArborX_DBSCANVerification.hpp#261*

### verifyDBSCAN

*_Bool verifyDBSCAN(ExecutionSpace exec_space, const Primitives & primitives, float eps, int core_min_size, const LabelsView & labels)*

*Defined at examples/dbscan/ArborX_DBSCANVerification.hpp#285*

### operator<<

*std::ostream & operator<<(std::ostream & os, const class ArborX::Point & p)*

*Defined at src/details/ArborX_DetailsTreeVisualization.hpp#27*

### project

*float project(const class ArborX::Point & p, const struct ArborX::Details::Direction & d)*

*Defined at src/details/ArborX_KDOP.hpp#127*



## Enums

| enum BufferStatus |

--

| PreallocationNone |
| PreallocationHard |
| PreallocationSoft |


*Defined at src/details/ArborX_DetailsCrsGraphWrapperImpl.hpp#27*



