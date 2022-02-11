# struct CountUpToN_DenseBox

*Defined at src/details/ArborX_DetailsFDBSCANDenseBox.hpp#83*

## Members

public Kokkos::View<int *, MemorySpace> _counts

public Primitives _primitives

public DenseCellOffsets _dense_cell_offsets

public int _num_dense_cells

public Permutation _permute

public int core_min_size

public float eps

public int _n



## Functions

### CountUpToN_DenseBox<MemorySpace, Primitives, DenseCellOffsets, Permutation>

*public void CountUpToN_DenseBox<MemorySpace, Primitives, DenseCellOffsets, Permutation>(const Kokkos::View<int *, MemorySpace> & counts, const Primitives & primitives, const DenseCellOffsets & dense_cell_offsets, const Permutation & permute, int core_min_size_in, float eps_in, int n)*

*Defined at src/details/ArborX_DetailsFDBSCANDenseBox.hpp#94*

### operator()

*public auto operator()(const Query & query, int k)*

*Defined at src/details/ArborX_DetailsFDBSCANDenseBox.hpp#111*



