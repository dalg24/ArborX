# struct KDOP

*Defined at src/details/ArborX_KDOP.hpp#142*

Inherits from Details::KDOP_Directions<k>



## Members

public Kokkos::Array<float, n_directions> _min_values

public Kokkos::Array<float, n_directions> _max_values



## Functions

### KDOP<k>

*public void KDOP<k>()*

*Defined at src/details/ArborX_KDOP.hpp#147*

### operator+=

*public KDOP<k> & operator+=(const class ArborX::Point & p)*

*Defined at src/details/ArborX_KDOP.hpp#155*

### operator+=

*public KDOP<k> & operator+=(const struct ArborX::Box & b)*

*Defined at src/details/ArborX_KDOP.hpp#167*

### operator+=

*public KDOP<k> & operator+=(const KDOP<k> & other)*

*Defined at src/details/ArborX_KDOP.hpp#203*

### operator Box

*public Box operator Box()*

*Defined at src/details/ArborX_KDOP.hpp#214*

### intersects

*public _Bool intersects(const class ArborX::Point & point)*

*Defined at src/details/ArborX_KDOP.hpp#225*

### intersects

*public _Bool intersects(const struct ArborX::Box & box)*

*Defined at src/details/ArborX_KDOP.hpp#237*

### intersects

*public _Bool intersects(const KDOP<k> & other)*

*Defined at src/details/ArborX_KDOP.hpp#243*



