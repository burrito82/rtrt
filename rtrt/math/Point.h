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
    RTRTDH Point(float x_ = 0.0f, float y_ = 0.0f, float z_ = 0.0f):
        Float4{x_, y_, z_, 1.0f}
    {
    }

    RTRTDHL explicit Point(Float4 const &f4):
        Float4{f4}
    {

    }
};

// add offset
RTRTDHLAPI Point const operator+(Point const &p, Vector const &v);
RTRTDHLAPI Point const operator+(Vector const &v, Point const &p);
RTRTDHLAPI Point &operator+=(Point &p, Vector const &v);

// sub offset
RTRTDHLAPI Point const operator-(Point const &p, Vector const &v);
RTRTDHLAPI Point const operator-(Vector const &v, Point const &p);
RTRTDHLAPI Point &operator-=(Point &p, Vector const &v);

// get offset
RTRTDHLAPI Vector const operator-(Point const &from, Point const &to);
RTRTDHLAPI float DistanceSquared(Point const &lhs, Point const &rhs);
RTRTDHLAPI float Distance(Point const &lhs, Point const &rhs);

} // namespace rtrt

#include "Point.inl"

#endif // ! RTRT_MATH_POINT_H
