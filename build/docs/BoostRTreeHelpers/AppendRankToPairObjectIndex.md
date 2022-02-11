# class AppendRankToPairObjectIndex

*Defined at test/ArborX_BoostRTreeHelpers.hpp#70*

 NOTE: Boost.Config defines BOOST_NO_CXX11_VARIADIC_TEMPLATES for nvcc with the current version of CUDA we are using.  In consequence we are not able to use std::tuple<Indexable, ...> which is unfortunate :(



## Members

private int _rank



## Functions

### AppendRankToPairObjectIndex

*public void AppendRankToPairObjectIndex(int rank)*

*Defined at test/ArborX_BoostRTreeHelpers.hpp#73*

### operator()

*public boost::tuple<T1, T2, int> operator()(const std::pair<T1, T2> & p)*

*Defined at test/ArborX_BoostRTreeHelpers.hpp#79*



