#ifndef RTRT_POINT_H
#define RTRT_POINT_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Vector.h"

#include <vector_types.h>
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

struct RTRTAPI Point : public float4
{
    Point(float x_ = 0.0f, float y_ = 0.0f, float z_ = 0.0f)
    {
        x = x_;
        y = y_;
        z = z_;
        w = 1.0f;
    }
};

// add offset
Point const operator+(Point const &p, Vector const &v)
{
    return Point
    {
        p.x + v.x,
        p.y + v.y,
        p.z + v.z
    };
}

Point const operator+(Vector const &v, Point const &p)
{
    return Point
    {
        v.x + p.x,
        v.y + p.y,
        v.z + p.z
    };
}

Point &operator+=(Point &p, Vector const &v)
{
    p.x += v.x;
    p.y += v.y;
    p.z += v.z;
    return p;
}

// sub offset
Point const operator-(Point const &p, Vector const &v)
{
    return Point
    {
        p.x - v.x,
        p.y - v.y,
        p.z - v.z
    };
}

Point const operator-(Vector const &v, Point const &p)
{
    return Point
    {
        v.x - p.x,
        v.y - p.y,
        v.z - p.z
    };
}

Point &operator-=(Point &p, Vector const &v)
{
    p.x -= v.x;
    p.y -= v.y;
    p.z -= v.z;
    return p;
}

// get offset
Vector const operator-(Point const &from, Point const &to)
{
    return Vector
    {
        from.x - to.x,
        from.y - to.y,
        from.z - to.z
    };
}

float DistanceSquared(Point const &lhs, Point const &rhs)
{
    return LengthSquared(lhs - rhs);
}

float Distance(Point const &lhs, Point const &rhs)
{
    return Length(lhs - rhs);
}

} // namespace rtrt

#endif // ! RTRT_POINT_H
