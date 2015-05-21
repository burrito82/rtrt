#include "Normal.h"

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

Normal const operator-(Normal const &n)
{
    return Normal{-n.x, -n.y, -n.z};
}

// addition
Normal const operator+(Normal const &lhs, Normal const &rhs)
{
    return Normal
    {
        lhs.x + rhs.x,
        lhs.y + rhs.y,
        lhs.z + rhs.z
    };
}

Normal &operator+=(Normal &lhs, Normal const &rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    lhs.Normalize();
    return lhs;
}

// subtraction
Normal const operator-(Normal const &lhs, Normal const &rhs)
{
    return Normal
    {
        lhs.x - rhs.x,
        lhs.y - rhs.y,
        lhs.z - rhs.z
    };
}

Normal &operator-=(Normal &lhs, Normal const &rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    lhs.Normalize();
    return lhs;
}

// scaling
Vector const operator*(float lhs, Normal const &rhs)
{
    return Vector
    {
        lhs * rhs.x,
        lhs * rhs.y,
        lhs * rhs.z
    };
}

Vector const operator*(Normal const &lhs, float rhs)
{
    return Vector
    {
        lhs.x * rhs,
        lhs.y * rhs,
        lhs.z * rhs
    };
}

// inv scaling by division
Vector const operator/(Normal const &lhs, float rhs)
{
    return (1.0f / rhs) * lhs;
}

float Dot(Normal const &lhs, Normal const &rhs)
{
    return lhs.x * rhs.x
        + lhs.y * rhs.y
        + lhs.z * rhs.z;
}

float Dot(Vector const &lhs, Normal const &rhs)
{
    return lhs.x * rhs.x
        + lhs.y * rhs.y
        + lhs.z * rhs.z;
}

float Dot(Normal const &lhs, Vector const &rhs)
{
    return lhs.x * rhs.x
        + lhs.y * rhs.y
        + lhs.z * rhs.z;
}

float AbsDot(Normal const &lhs, Normal const &rhs)
{
    return std::abs(Dot(lhs, rhs));
}

float AbsDot(Vector const &lhs, Normal const &rhs)
{
    return std::abs(Dot(lhs, rhs));
}

float AbsDot(Normal const &lhs, Vector const &rhs)
{
    return std::abs(Dot(lhs, rhs));
}

Normal FaceForward(Normal const &n, Vector const &v)
{
    return (Dot(n, v) > 0.0f ? n : -n);
}

} // namespace rtrt

