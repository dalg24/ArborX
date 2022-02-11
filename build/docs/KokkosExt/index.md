# namespace KokkosExt



## Namespaces

* [ArithmeticTraits](ArithmeticTraits/index.md)


## Records

* [is_accessible_from](is_accessible_from.md)
* [is_accessible_from](is_accessible_from.md)
* [is_accessible_from_host](is_accessible_from_host.md)
* [ScopedProfileRegion](ScopedProfileRegion.md)


## Functions

### max

*const T & max(const T & a, const T & b)*

*Defined at src/details/ArborX_DetailsKokkosExtMinMaxOperations.hpp#24*

 Compute the maximum of two values.

### min

*const T & min(const T & a, const T & b)*

*Defined at src/details/ArborX_DetailsKokkosExtMinMaxOperations.hpp#31*

 Compute the minimum of two values.

### max

*T max(std::initializer_list<T> ilist)*

*Defined at src/details/ArborX_DetailsKokkosExtMinMaxOperations.hpp#37*

### min

*T min(std::initializer_list<T> ilist)*

*Defined at src/details/ArborX_DetailsKokkosExtMinMaxOperations.hpp#57*

### isFinite

*_Bool isFinite(T x)*

*Defined at src/details/ArborX_DetailsKokkosExtMathFunctions.hpp#30*

 Determine whether the given floating point argument 

**x**

 NOTE: Clang issues a warning if the std:: namespace is missing and nvcc complains about calling a __host__ function from a __host__ __device__ function when it is present.

### version

*basic_string version()*

*Defined at src/details/ArborX_DetailsKokkosExtVersion.hpp#23*

### swap

*void swap(T & a, T & b)*

*Defined at src/details/ArborX_DetailsKokkosExtSwap.hpp#24*



