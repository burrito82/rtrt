#include "Point.h"

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/

/*============================================================================*/
/* MACROS AND DEFINES, CONSTANTS AND STATICS                                  */
/*============================================================================*/
namespace rtrt
{
/*============================================================================*/
/* LOCAL VARS AND FUNCS                                                       */
/*============================================================================*/

/*============================================================================*/
/* CONSTRUCTORS / DESTRUCTOR                                                  */
/*============================================================================*/

/*============================================================================*/
/* IMPLEMENTATION                                                             */
/*============================================================================*/

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

