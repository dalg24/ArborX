# class Boxes

*Defined at examples/simple_intersection/example_intersection.cpp#33*

 Perform intersection queries using the same objects for the queries as the objects used in BVH construction that are located on a regular spaced three-dimensional grid. Each box will only intersect with itself.

 i-2  i-1  i  i+1

  o    o   o   o   j+1          ---  o    o | x | o   j          ---  o    o   o   o   j-1

  o    o   o   o   j-2



## Members

private Kokkos::View<ArborX::Box *, typename DeviceType::memory_space> _boxes



## Functions

### Boxes<DeviceType>

*public void Boxes<DeviceType>(const typename DeviceType::execution_space & execution_space)*

*Defined at examples/simple_intersection/example_intersection.cpp#38*

 Create non-intersecting boxes on a 3D cartesian grid used both for queries and predicates.

### size

*public int size()*

*Defined at examples/simple_intersection/example_intersection.cpp#73*

 Return the number of boxes.

### get_box

*public const ArborX::Box & get_box(int i)*

*Defined at examples/simple_intersection/example_intersection.cpp#76*

 Return the box with index i.



