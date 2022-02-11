# namespace CrsGraphWrapperImpl



## Records

* [Iota](Iota.md)


## Functions

### queryImpl

*void queryImpl(const ExecutionSpace & space, const Tree & tree, const Predicates & predicates, const Callback & callback, OutputView & out, OffsetView & offset, PermuteType permute, enum ArborX::Details::BufferStatus buffer_status)*

*Defined at src/details/ArborX_DetailsCrsGraphWrapperImpl.hpp#134*

### allocateAndInitializeStorage

*std::enable_if_t<std::is_same<Tag, SpatialPredicateTag>({})> allocateAndInitializeStorage(Tag , const ExecutionSpace & space, const Predicates & predicates, OffsetView & offset, OutView & out, int buffer_size)*

*Defined at src/details/ArborX_DetailsCrsGraphWrapperImpl.hpp#310*

### allocateAndInitializeStorage

*std::enable_if_t<std::is_same<Tag, NearestPredicateTag>({})> allocateAndInitializeStorage(Tag , const ExecutionSpace & space, const Predicates & predicates, OffsetView & offset, OutView & out, int )*

*Defined at src/details/ArborX_DetailsCrsGraphWrapperImpl.hpp#336*

### queryDispatch

*std::enable_if_t<!is_tagged_post_callback<Callback>({}) && Kokkos::is_view<OutputView>({}) && Kokkos::is_view<OffsetView>({})> queryDispatch(Tag , const Tree & tree, const ExecutionSpace & space, const Predicates & predicates, const Callback & callback, OutputView & out, OffsetView & offset, const Experimental::TraversalPolicy & policy)*

*Defined at src/details/ArborX_DetailsCrsGraphWrapperImpl.hpp#361*

### queryDispatch

*std::enable_if_t<Kokkos::is_view<Indices>({}) && Kokkos::is_view<Offset>({})> queryDispatch(Tag , const Tree & tree, const ExecutionSpace & space, const Predicates & predicates, Indices & indices, Offset & offset, const Experimental::TraversalPolicy & policy)*

*Defined at src/details/ArborX_DetailsCrsGraphWrapperImpl.hpp#414*

### queryDispatch

*std::enable_if_t<is_tagged_post_callback<Callback>({})> queryDispatch(Tag , const Tree & tree, const ExecutionSpace & space, const Predicates & predicates, const Callback & callback, OutputView & out, OffsetView & offset, const Experimental::TraversalPolicy & policy)*

*Defined at src/details/ArborX_DetailsCrsGraphWrapperImpl.hpp#427*

### check_valid_callback_if_first_argument_is_not_a_view

*std::enable_if_t<!Kokkos::is_view<Callback>({}) && !is_tagged_post_callback<Callback>({})> check_valid_callback_if_first_argument_is_not_a_view(const Callback & callback, const Predicates & predicates, const OutputView & out)*

*Defined at src/details/ArborX_DetailsCrsGraphWrapperImpl.hpp#442*

### check_valid_callback_if_first_argument_is_not_a_view

*std::enable_if_t<!Kokkos::is_view<Callback>({}) && is_tagged_post_callback<Callback>({})> check_valid_callback_if_first_argument_is_not_a_view(const Callback & , const Predicates & , const OutputView & )*

*Defined at src/details/ArborX_DetailsCrsGraphWrapperImpl.hpp#452*

### check_valid_callback_if_first_argument_is_not_a_view

*std::enable_if_t<Kokkos::is_view<View>({})> check_valid_callback_if_first_argument_is_not_a_view(const View & , const Predicates & , const OutputView & )*

*Defined at src/details/ArborX_DetailsCrsGraphWrapperImpl.hpp#462*



