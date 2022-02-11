# class BasicBoundingVolumeHierarchy

*Defined at src/ArborX_LinearBVH.hpp#41*

## Members

private size_t _size

private ArborX::BasicBoundingVolumeHierarchy::bounding_volume_type _bounds

private Kokkos::View<node_type *, MemorySpace> _internal_and_leaf_nodes



## Functions

### BasicBoundingVolumeHierarchy<MemorySpace, BoundingVolume, Enable>

*public void BasicBoundingVolumeHierarchy<MemorySpace, BoundingVolume, Enable>()*

*Defined at src/ArborX_LinearBVH.hpp#49*

### BasicBoundingVolumeHierarchy<MemorySpace, BoundingVolume, Enable>

*public void BasicBoundingVolumeHierarchy<MemorySpace, BoundingVolume, Enable>(const ExecutionSpace & space, const Primitives & primitives)*

*Defined at src/ArborX_LinearBVH.hpp#192*

### size

*public ArborX::BasicBoundingVolumeHierarchy::size_type size()*

*Defined at src/ArborX_LinearBVH.hpp#56*

### empty

*public _Bool empty()*

*Defined at src/ArborX_LinearBVH.hpp#59*

### bounds

*public ArborX::BasicBoundingVolumeHierarchy::bounding_volume_type bounds()*

*Defined at src/ArborX_LinearBVH.hpp#62*

### query

*public void query(const ExecutionSpace & space, const Predicates & predicates, const Callback & callback, const Experimental::TraversalPolicy & policy)*

*Defined at src/ArborX_LinearBVH.hpp#275*

### query

*public std::enable_if_t<Kokkos::is_view<std::decay_t<View> >({})> query(const ExecutionSpace & space, const Predicates & predicates, CallbackOrView && callback_or_view, View && view, Args &&... args)*

*Defined at src/ArborX_LinearBVH.hpp#72*

### getInternalNodes

*private Kokkos::View<node_type *, MemorySpace> getInternalNodes()*

*Defined at src/ArborX_LinearBVH.hpp#101*

### getLeafNodes

*private Kokkos::View<node_type *, MemorySpace> getLeafNodes()*

*Defined at src/ArborX_LinearBVH.hpp#108*

### getLeafNodes

*private Kokkos::View<const node_type *, MemorySpace> getLeafNodes()*

*Defined at src/ArborX_LinearBVH.hpp#114*

### getRootBoundingVolumePtr

*private const ArborX::BasicBoundingVolumeHierarchy::bounding_volume_type * getRootBoundingVolumePtr()*

*Defined at src/ArborX_LinearBVH.hpp#122*

### BasicBoundingVolumeHierarchy<type-parameter-0-0, typename enable_if<Kokkos::is_device<DeviceType>::value, void>::type, void>

*public void BasicBoundingVolumeHierarchy<type-parameter-0-0, typename enable_if<Kokkos::is_device<DeviceType>::value, void>::type, void>()*

*Defined at src/ArborX_LinearBVH.hpp#151*

 clang-format off

### BasicBoundingVolumeHierarchy<type-parameter-0-0, typename enable_if<Kokkos::is_device<DeviceType>::value, void>::type, void>

*public void BasicBoundingVolumeHierarchy<type-parameter-0-0, typename enable_if<Kokkos::is_device<DeviceType>::value, void>::type, void>(const Primitives & primitives)*

*Defined at src/ArborX_LinearBVH.hpp#155*

### query

*public std::enable_if_t<!Kokkos::is_execution_space<FirstArgumentType>::value> query(FirstArgumentType && arg1, Args &&... args)*

*Defined at src/ArborX_LinearBVH.hpp#162*

 clang-format on

### query

*private std::enable_if_t<Kokkos::is_execution_space<FirstArgumentType>::value> query(const FirstArgumentType & space, Args &&... args)*

*Defined at src/ArborX_LinearBVH.hpp#179*



