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
    Point(float x_ = 0.0f, float y_ = 0.0f, float z_ = 0.0f):
        Float4{x_, y_, z_, 1.0f}
    {
    }

    explicit Point(Float4 const &f4):
        Float4{f4}
    {

    }
};

// add offset
RTRTAPI Point const operator+(Point const &p, Vector const &v);
RTRTAPI Point const operator+(Vector const &v, Point const &p);
RTRTAPI Point &operator+=(Point &p, Vector const &v);

// sub offset
RTRTAPI Point const operator-(Point const &p, Vector const &v);
RTRTAPI Point const operator-(Vector const &v, Point const &p);
RTRTAPI Point &operator-=(Point &p, Vector const &v);

// get offset
RTRTAPI Vector const operator-(Point const &from, Point const &to);
RTRTAPI float DistanceSquared(Point const &lhs, Point const &rhs);
RTRTAPI float Distance(Point const &lhs, Point const &rhs);

} // namespace rtrt

#endif // ! RTRT_MATH_POINT_H
