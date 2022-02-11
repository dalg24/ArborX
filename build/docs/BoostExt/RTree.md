# class RTree

*Defined at test/ArborX_BoostRTreeHelpers.hpp#245*

 FIXME Goal is to match the BVH interface



## Members

private BoostRTreeHelpers::RTree<Indexable> _tree



## Functions

### RTree<Indexable>

*public void RTree<Indexable>(ExecutionSpace , const Kokkos::View<Indexable *, DeviceType> & values)*

*Defined at test/ArborX_BoostRTreeHelpers.hpp#252*

### query

*public void query(const ExecutionSpace & , const Predicates & predicates, InputView & indices, InputView & offset, TrailingArgs &&... )*

*Defined at test/ArborX_BoostRTreeHelpers.hpp#262*

 WARNING trailing pack will match anything :/

### query

*public void query(const ExecutionSpace & , const Predicates & , const Callback & , TrailingArgs &&... )*

*Defined at test/ArborX_BoostRTreeHelpers.hpp#273*



