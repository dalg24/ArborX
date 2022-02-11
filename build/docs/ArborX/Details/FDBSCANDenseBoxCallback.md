# struct FDBSCANDenseBoxCallback

*Defined at src/details/ArborX_DetailsFDBSCANDenseBox.hpp#150*

## Members

public UnionFind<MemorySpace> _union_find

public CorePointsType _is_core_point

public Primitives _primitives

public DenseCellOffsets _dense_cell_offsets

public int _num_dense_cells

public int _num_points_in_dense_cells

public Permutation _permute

public float eps



## Functions

### FDBSCANDenseBoxCallback<MemorySpace, CorePointsType, Primitives, DenseCellOffsets, Permutation>

*public void FDBSCANDenseBoxCallback<MemorySpace, CorePointsType, Primitives, DenseCellOffsets, Permutation>(const Kokkos::View<int *, MemorySpace> & labels, const CorePointsType & is_core_point, const Primitives & primitives, const DenseCellOffsets & dense_cell_offsets, const Permutation & permute, float eps_in)*

*Defined at src/details/ArborX_DetailsFDBSCANDenseBox.hpp#161*

### operator()

*public auto operator()(const Query & query, int k)*

*Defined at src/details/ArborX_DetailsFDBSCANDenseBox.hpp#178*



