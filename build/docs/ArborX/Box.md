# struct Box

*Defined at src/details/ArborX_Box.hpp#28*

 Axis-Aligned Bounding Box. This is just a thin wrapper around an array of size 2x spatial dimension with a default constructor to initialize properly an "empty" box.



## Members

Point _min_corner

Point _max_corner



## Functions

### Box

*public void Box()*

*Defined at src/details/ArborX_Box.hpp#30*

### Box

*public void Box(const class ArborX::Point & min_corner, const class ArborX::Point & max_corner)*

*Defined at src/details/ArborX_Box.hpp#33*

### minCorner

*public class ArborX::Point & minCorner()*

*Defined at src/details/ArborX_Box.hpp#40*

### minCorner

*public const class ArborX::Point & minCorner()*

*Defined at src/details/ArborX_Box.hpp#43*

### minCorner

*public volatile class ArborX::Point & minCorner()*

*Defined at src/details/ArborX_Box.hpp#46*

### minCorner

*public const volatile class ArborX::Point & minCorner()*

*Defined at src/details/ArborX_Box.hpp#49*

### maxCorner

*public class ArborX::Point & maxCorner()*

*Defined at src/details/ArborX_Box.hpp#52*

### maxCorner

*public const class ArborX::Point & maxCorner()*

*Defined at src/details/ArborX_Box.hpp#55*

### maxCorner

*public volatile class ArborX::Point & maxCorner()*

*Defined at src/details/ArborX_Box.hpp#58*

### maxCorner

*public const volatile class ArborX::Point & maxCorner()*

*Defined at src/details/ArborX_Box.hpp#61*

### operator+=

*public struct ArborX::Box & operator+=(const struct ArborX::Box & other)*

*Defined at src/details/ArborX_Box.hpp#71*

### operator+=

*public void operator+=(const volatile struct ArborX::Box & other)*

*Defined at src/details/ArborX_Box.hpp#84*

### operator+=

*public struct ArborX::Box & operator+=(const class ArborX::Point & point)*

*Defined at src/details/ArborX_Box.hpp#96*



