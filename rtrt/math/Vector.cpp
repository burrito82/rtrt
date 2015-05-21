#include "Vector.h"

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

Vector const operator-(Vector const &n)
{
    return Vector{-n.x, -n.y, -n.z};
}

// addition
Vector const operator+(Vector const &lhs, Vector const &rhs)
{
    return Vector
    {
        lhs.x + rhs.x,
        lhs.y + rhs.y,
        lhs.z + rhs.z
    };
}

Vector &operator+=(Vector &lhs, Vector const &rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}

// subtraction
Vector const operator-(Vector const &lhs, Vector const &rhs)
{
    return Vector
    {
        lhs.x - rhs.x,
        lhs.y - rhs.y,
        lhs.z - rhs.z
    };
}

Vector &operator-=(Vector &lhs, Vector const &rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    return lhs;
}

// scaling
Vector const operator*(float lhs, Vector const &rhs)
{
    return Vector
    {
        lhs * rhs.x,
        lhs * rhs.y,
        lhs * rhs.z
    };
}

Vector const operator*(Vector const &lhs, float rhs)
{
    return Vector
    {
        lhs.x * rhs,
        lhs.y * rhs,
        lhs.z * rhs
    };
}

Vector &operator*=(Vector &lhs, float rhs)
{
    lhs.x *= rhs;
    lhs.y *= rhs;
    lhs.z *= rhs;
    return lhs;
}

// inv scaling by division
Vector const operator/(Vector const &lhs, float rhs)
{
    return (1.0f / rhs) * lhs;
}

Vector &operator/=(Vector &lhs, float rhs)
{
    return lhs *= (1.0f / rhs);
}

float Dot(Vector const &lhs, Vector const &rhs)
{
    return lhs.x * rhs.x
        + lhs.y * rhs.y
        + lhs.z * rhs.z;
}

float AbsDot(Vector const &lhs, Vector const &rhs)
{
    return std::abs(Dot(lhs, rhs));
}

Vector const Cross(Vector const &lhs, Vector const &rhs)
{
    return Vector
    {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x
    };
}

float LengthSquared(Vector const &v)
{
    return Dot(v, v);
}

float Length(Vector const &v)
{
    return std::sqrt(LengthSquared(v));
}

Vector &Normalize(Vector &v)
{
    return v /= Length(v);
}

Vector const Normalized(Vector v)
{
    return Normalize(v);
}

/*============================================================================*/
/* OTHER HELPER FUNCTIONS                                                     */
/*============================================================================*/

void CoordinateSystem(Vector const &v0, Vector &v1, Vector &v2)
{
    if (std::abs(v0.x) > std::abs(v0.y))
    {
        float fInvLen = 1.0f / std::sqrt(v0.x * v0.x + v0.z * v0.z);
        v1 = Vector{fInvLen * (-v0.z), 0.0f, fInvLen * v0.x};
    }
    else
    {
        float fInvLen = 1.0f / std::sqrt(v0.z * v0.z + v0.y * v0.y);
        v1 = Vector{0.0f, fInvLen * v0.z, fInvLen * (-v0.y)};
    }
    v2 = Cross(v0, v1);
}

} // namespace rtrt

