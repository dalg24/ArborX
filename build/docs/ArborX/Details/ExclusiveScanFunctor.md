# class ExclusiveScanFunctor

*Defined at src/details/ArborX_DetailsUtils.hpp#148*

 NOTE: This functor is used in exclusivePrefixSum( src, dst ).  We were getting a compile error on CUDA when using a KOKKOS_LAMBDA.



## Members

private Kokkos::View<T *, DeviceType> _in

private Kokkos::View<T *, DeviceType> _out



## Functions

### ExclusiveScanFunctor<T, DeviceType>

*public void ExclusiveScanFunctor<T, DeviceType>(const Kokkos::View<T *, DeviceType> & in, const Kokkos::View<T *, DeviceType> & out)*

*Defined at src/details/ArborX_DetailsUtils.hpp#151*

### operator()

*public void operator()(int i, T & update, _Bool final_pass)*

*Defined at src/details/ArborX_DetailsUtils.hpp#157*



