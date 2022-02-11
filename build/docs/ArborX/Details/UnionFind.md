# struct UnionFind

*Defined at src/details/ArborX_DetailsUnionFind.hpp#76*

## Members

public Kokkos::View<int *, MemorySpace> labels_



## Functions

### UnionFind<MemorySpace>

*public void UnionFind<MemorySpace>(Kokkos::View<int *, MemorySpace> labels)*

*Defined at src/details/ArborX_DetailsUnionFind.hpp#80*

### representative

*public int representative(const int i)*

*Defined at src/details/ArborX_DetailsUnionFind.hpp#109*

 Per [1]:

 Note that the [`representative()`] code is re-entrant and synchronization free even though concurrent execution may cause data races on the parent array. However, these races are guaranteed to be benign for the following reasons. First, the only write to shared data is in [`parent[prev] = next; `]. This write updates a single aligned machine word and is therefore atomic. Moreover, it overwrites a valid entry with another valid entry. Hence, it does not matter if other threads see the old or the new value as either value will allow them to eventually reach the representative. Similarly, all the reads of the parent array will either fetch the old or new value, but both values are acceptable. The only problem that can occur is that two threads try to update the same parent pointer at the same time. In this case, one of the updates is lost. This reduces the code’s performance as duplicate work is performed and the path is not shortened by as much as it could have been, but it does not result in incorrect paths. On average, the savings of not having to perform synchronization far outweighs this small cost. Lastly, it should be noted that the rest of the code either accesses the parent array via calls to the find_repres function or changes the parent pointer of a representative vertex but never of a vertex that is in the middle of a path. If the find_repres code already sees the new representative, it will return it. Otherwise, it will return the old representative. Either return value is handled correctly.

### merge_into

*public void merge_into(int i, int j)*

*Defined at src/details/ArborX_DetailsUnionFind.hpp#132*

 In some situations it is necessary to make sure that the a particular label is assigned to a point. As a regular merge() does not guarantee that, an extra function is introduced, which assigns the label of the second point (or, rather, the label of its representative) to the first.

### merge

*public void merge(int i, int j)*

*Defined at src/details/ArborX_DetailsUnionFind.hpp#135*



