# class BruteForce

*Defined at src/ArborX_BruteForce.hpp#27*

## Members

private ArborX::BruteForce::size_type _size

Box _bounds

private Kokkos::View<bounding_volume_type *, memory_space> _bounding_volumes



## Functions

### BruteForce<MemorySpace>

*public void BruteForce<MemorySpace>()*

*Defined at src/ArborX_BruteForce.hpp#35*

### BruteForce<MemorySpace>

*public void BruteForce<MemorySpace>(const ExecutionSpace & space, const Primitives & primitives)*

*Defined at src/ArborX_BruteForce.hpp#71*

### size

*public ArborX::BruteForce::size_type size()*

*Defined at src/ArborX_BruteForce.hpp#41*

### empty

*public _Bool empty()*

*Defined at src/ArborX_BruteForce.hpp#44*

### bounds

*public Box bounds()*

*Defined at src/ArborX_BruteForce.hpp#47*

### query

*public void query(const ExecutionSpace & space, const Predicates & predicates, const Callback & callback, Ignore )*

*Defined at src/ArborX_BruteForce.hpp#97*

### query

*public std::enable_if_t<Kokkos::is_view<std::decay_t<View> >({})> query(const ExecutionSpace & space, const Predicates & predicates, CallbackOrView && callback_or_view, View && view, Args &&... args)*

*Defined at src/ArborX_BruteForce.hpp#56*



