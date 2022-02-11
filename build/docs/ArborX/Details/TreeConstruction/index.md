# namespace TreeConstruction



## Namespaces

* [@nonymous_namespace](@nonymous_namespace/index.md)


## Records

* [GenerateHierarchy](GenerateHierarchy.md)


## Functions

### calculateBoundingBoxOfTheScene

*void calculateBoundingBoxOfTheScene(const ExecutionSpace & space, const Primitives & primitives, struct ArborX::Box & scene_bounding_box)*

*Defined at src/details/ArborX_DetailsTreeConstruction.hpp#42*

### assignMortonCodesImpl

*std::enable_if_t<std::is_same<Box, typename AccessTraitsHelper<AccessTraits<Primitives, PrimitivesTag> >::type>::value> assignMortonCodesImpl(const ExecutionSpace & space, const Primitives & primitives, MortonCodes morton_codes, const struct ArborX::Box & scene_bounding_box)*

*Defined at src/details/ArborX_DetailsTreeConstruction.hpp#58*

### assignMortonCodesImpl

*std::enable_if_t<std::is_same<Point, typename AccessTraitsHelper<AccessTraits<Primitives, PrimitivesTag> >::type>::value> assignMortonCodesImpl(const ExecutionSpace & space, const Primitives & primitives, MortonCodes morton_codes, const struct ArborX::Box & scene_bounding_box)*

*Defined at src/details/ArborX_DetailsTreeConstruction.hpp#77*

### assignMortonCodes

*void assignMortonCodes(const ExecutionSpace & space, const Primitives & primitives, Kokkos::View<unsigned int *, MortonCodesViewProperties...> morton_codes, const struct ArborX::Box & scene_bounding_box)*

*Defined at src/details/ArborX_DetailsTreeConstruction.hpp#96*

### initializeSingleLeafNode

*void initializeSingleLeafNode(const ExecutionSpace & space, const Primitives & primitives, const Nodes & leaf_nodes)*

*Defined at src/details/ArborX_DetailsTreeConstruction.hpp#110*

### generateHierarchy

*void generateHierarchy(const ExecutionSpace & space, const Primitives & primitives, Kokkos::View<unsigned int *, PermutationIndicesViewProperties...> permutation_indices, Kokkos::View<unsigned int *, MortonCodesViewProperties...> sorted_morton_codes, Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes, Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes)*

*Defined at src/details/ArborX_DetailsTreeConstruction.hpp#396*



