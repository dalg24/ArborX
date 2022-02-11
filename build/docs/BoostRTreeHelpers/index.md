# namespace BoostRTreeHelpers



## Records

* [PairMaker](PairMaker.md)
* [AppendRankToPairObjectIndex](AppendRankToPairObjectIndex.md)
* [UnaryPredicate](UnaryPredicate.md)


## Functions

### makeRTree

*RTree<typename View::value_type> makeRTree(const View & objects)*

*Defined at test/ArborX_BoostRTreeHelpers.hpp#59*

### translate

*auto translate(const ArborX::Intersects<ArborX::Sphere> & query)*

*Defined at test/ArborX_BoostRTreeHelpers.hpp#152*

### translate

*auto translate(const ArborX::Intersects<ArborX::Box> & query)*

*Defined at test/ArborX_BoostRTreeHelpers.hpp#169*

### translate

*auto translate(const ArborX::Nearest<Geometry> & query)*

*Defined at test/ArborX_BoostRTreeHelpers.hpp#176*

### performQueries

*std::tuple<OutputView, OutputView> performQueries(const RTree<Indexable> & rtree, const InputView & queries)*

*Defined at test/ArborX_BoostRTreeHelpers.hpp#185*



