# class GenerateHierarchy

*Defined at src/details/ArborX_DetailsTreeConstruction.hpp#143*

## Members

private Primitives _primitives

private Kokkos::View<const unsigned int *, MemorySpace> _permutation_indices

private Kokkos::View<const unsigned int *, MemorySpace> _sorted_morton_codes

private Kokkos::View<Node *, MemorySpace> _leaf_nodes

private Kokkos::View<Node *, MemorySpace> _internal_nodes

private Kokkos::View<int *, MemorySpace> _ranges

private int _num_internal_nodes



## Functions

### GenerateHierarchy<Primitives, MemorySpace, Node>

*public void GenerateHierarchy<Primitives, MemorySpace, Node>(const ExecutionSpace & space, const Primitives & primitives, Kokkos::View<const unsigned int *, PermutationIndicesViewProperties...> permutation_indices, Kokkos::View<const unsigned int *, MortonCodesViewProperties...> sorted_morton_codes, Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes, Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes)*

*Defined at src/details/ArborX_DetailsTreeConstruction.hpp#151*

### delta

*public int delta(const int i)*

*Defined at src/details/ArborX_DetailsTreeConstruction.hpp#179*

### getNodePtr

*public Node * getNodePtr(int i)*

*Defined at src/details/ArborX_DetailsTreeConstruction.hpp#211*

### setRightChild

*public std::enable_if_t<std::is_same<Tag, NodeWithTwoChildrenTag>({})> setRightChild(Node * node, int child_right)*

*Defined at src/details/ArborX_DetailsTreeConstruction.hpp#218*

### setRightChild

*public std::enable_if_t<std::is_same<Tag, NodeWithLeftChildAndRopeTag>({})> setRightChild(Node * node, int )*

*Defined at src/details/ArborX_DetailsTreeConstruction.hpp#227*

### setRope

*public std::enable_if_t<std::is_same<Tag, NodeWithTwoChildrenTag>({})> setRope(Node * , int , int )*

*Defined at src/details/ArborX_DetailsTreeConstruction.hpp#235*

### setRope

*public std::enable_if_t<std::is_same<Tag, NodeWithLeftChildAndRopeTag>({})> setRope(Node * node, int range_right, int delta_right)*

*Defined at src/details/ArborX_DetailsTreeConstruction.hpp#242*

### operator()

*public void operator()(int i)*

*Defined at src/details/ArborX_DetailsTreeConstruction.hpp#266*



