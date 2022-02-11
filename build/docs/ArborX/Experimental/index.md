# namespace Experimental



## Records

* [TraversalPolicy](TraversalPolicy.md)
* [KDOP](KDOP.md)
* [Vector](Vector.md)
* [Ray](Ray.md)


## Functions

### expand

*void expand(KDOP<k> & that, const KDOP<k> & other)*

*Defined at src/details/ArborX_KDOP.hpp#258*

### expand

*void expand(KDOP<k> & that, const class ArborX::Point & point)*

*Defined at src/details/ArborX_KDOP.hpp#264*

### expand

*void expand(KDOP<k> & that, const struct ArborX::Box & box)*

*Defined at src/details/ArborX_KDOP.hpp#270*

### expand

*void expand(struct ArborX::Box & a, const KDOP<k> & b)*

*Defined at src/details/ArborX_KDOP.hpp#276*

### intersects

*_Bool intersects(const struct ArborX::Box & a, const KDOP<k> & b)*

*Defined at src/details/ArborX_KDOP.hpp#283*

 NOTE intersects(predicate_geometry, bounding_volume)

### intersects

*_Bool intersects(const KDOP<k> & a, const struct ArborX::Box & b)*

*Defined at src/details/ArborX_KDOP.hpp#289*

### intersects

*_Bool intersects(const class ArborX::Point & p, const KDOP<k> & x)*

*Defined at src/details/ArborX_KDOP.hpp#295*

### intersects

*_Bool intersects(const KDOP<k> & a, const KDOP<k> & b)*

*Defined at src/details/ArborX_KDOP.hpp#301*

### returnCentroid

*Point returnCentroid(const KDOP<k> & p)*

*Defined at src/details/ArborX_KDOP.hpp#307*

### operator==

*_Bool operator==(const struct ArborX::Experimental::Vector & v, const struct ArborX::Experimental::Vector & w)*

*Defined at src/details/ArborX_Ray.hpp#35*

### makeVector

*Vector makeVector(const class ArborX::Point & begin, const class ArborX::Point & end)*

*Defined at src/details/ArborX_Ray.hpp#42*

### dotProduct

*float dotProduct(const struct ArborX::Experimental::Vector & v, const struct ArborX::Experimental::Vector & w)*

*Defined at src/details/ArborX_Ray.hpp#53*

### crossProduct

*Vector crossProduct(const struct ArborX::Experimental::Vector & v, const struct ArborX::Experimental::Vector & w)*

*Defined at src/details/ArborX_Ray.hpp#59*

### equals

*_Bool equals(const struct ArborX::Experimental::Vector & v, const struct ArborX::Experimental::Vector & w)*

*Defined at src/details/ArborX_Ray.hpp#66*

### equals

*_Bool equals(const struct ArborX::Experimental::Ray & l, const struct ArborX::Experimental::Ray & r)*

*Defined at src/details/ArborX_Ray.hpp#127*

### returnCentroid

*Point returnCentroid(const struct ArborX::Experimental::Ray & ray)*

*Defined at src/details/ArborX_Ray.hpp#134*

### intersection

*_Bool intersection(const struct ArborX::Experimental::Ray & ray, const struct ArborX::Box & box, float & tmin, float & tmax)*

*Defined at src/details/ArborX_Ray.hpp#158*

 The ray-box intersection algorithm is based on [1]. Their 'efficient slag' algorithm checks the intersections both in front and behind the ray.

 There are few issues here. First, when a ray direction is aligned with one of the axis, a division by zero will occur. This is fine, as usually it results in +inf or -inf, which are treated correctly. However, it also leads to the second situation, when it is 0/0 which occurs when the ray's origin in that dimension is on the same plane as one of the corners of the box (i.e., if inv_ray_dir[d] == 0 && (min_corner[d] == origin[d] || max_corner[d] == origin[d])). This leads to NaN, which are not treated correctly (unless, as in [1], the underlying min/max functions are able to ignore them). The issue is discussed in more details in [2] and the website (key word: A minimal ray-tracer: rendering simple shapes).

 [1] Majercik, A., Crassin, C., Shirley, P., & McGuire, M. (2018). A ray-box intersection algorithm and efficient dynamic voxel rendering. Journal of Computer Graphics Techniques Vol, 7(3).

 [2] Williams, A., Barrus, S., Morley, R. K., & Shirley, P. (2005). An efficient and robust ray-box intersection algorithm. In ACM SIGGRAPH 2005 Courses (pp. 9-es).

### intersects

*_Bool intersects(const struct ArborX::Experimental::Ray & ray, const struct ArborX::Box & box)*

*Defined at src/details/ArborX_Ray.hpp#192*

### solveQuadratic

*_Bool solveQuadratic(const float a, const float b, const float c, float & x1, float & x2)*

*Defined at src/details/ArborX_Ray.hpp#204*

 Solves a*x^2 + b*x + c = 0. If a solution exists, return true and stores roots at x1, x2. If a solution does not exist, returns false.

### intersection

*_Bool intersection(const struct ArborX::Experimental::Ray & ray, const struct ArborX::Sphere & sphere, float & tmin, float & tmax)*

*Defined at src/details/ArborX_Ray.hpp#248*

 Ray-Sphere intersection algorithm.

 The sphere can be expressed as the solution to     |p - c|^2 - r^2 = 0,           (1) where c is the center of the sphere, and r is the radius. On the other hand, any point on a bidirectional ray satisfies     p = o + t*d,                   (2) where o is the origin, and d is the direction vector. Substituting (2) into (1),     |(o + t*d) - c|^2 - r^2 = 0,   (3) results in a quadratic equation for unknown t     a2 * t^2 + a1 * t + a0 = 0 with     a2 = |d|^2, a1 = 2*(d, o - c), and a0 = |o - c|^2 - r^2. Then, we only need to intersect the solution interval [tmin, tmax] with [0, +inf) for the unidirectional ray.

### overlapDistance

*float overlapDistance(const struct ArborX::Experimental::Ray & ray, const struct ArborX::Sphere & sphere)*

*Defined at src/details/ArborX_Ray.hpp#274*



