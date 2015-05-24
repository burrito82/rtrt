#ifndef RTRT_MATH_POINT_H
#define RTRT_MATH_POINT_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Vector.h"

#include "../cuda/Float4.cuh"
/*============================================================================*/
/* DEFINES                                                                    */
/*============================================================================*/

/*============================================================================*/
/* FORWARD DECLARATIONS                                                       */
/*============================================================================*/
namespace rtrt
{
/*============================================================================*/
/* STRUCT DEFINITIONS                                                         */
/*============================================================================*/

struct RTRTAPI Point : public Float4
{
    __device__ __host__
    Point(float x_ = 0.0f, float y_ = 0.0f, float z_ = 0.0f):
        Float4{x_, y_, z_, 1.0f}
    {
    }

    __device__ __host__
    explicit Point(Float4 const &f4):
        Float4{f4}
    {

    }
};

#define RMPHL __device__ __host__ __inline__ RTRTAPI
// add offset
RMPHL Point const operator+(Point const &p, Vector const &v);
RMPHL Point const operator+(Vector const &v, Point const &p);
RMPHL Point &operator+=(Point &p, Vector const &v);

// sub offset
RMPHL Point const operator-(Point const &p, Vector const &v);
RMPHL Point const operator-(Vector const &v, Point const &p);
RMPHL Point &operator-=(Point &p, Vector const &v);

// get offset
RMPHL Vector const operator-(Point const &from, Point const &to);
RMPHL float DistanceSquared(Point const &lhs, Point const &rhs);
RMPHL float Distance(Point const &lhs, Point const &rhs);

#undef RMPHL

} // namespace rtrt

#include "Point.inl"

#endif // ! RTRT_MATH_POINT_H
