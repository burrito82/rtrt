#ifndef RTRT_NORMAL_H
#define RTRT_NORMAL_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Vector.h"
#include "../LibraryConfig.h"

#include <vector_types.h>
#include <cmath>
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

struct RTRTAPI Normal : public float4
{
    Normal(float x_, float y_, float z_)
    {
        x = x_;
        y = y_;
        z = z_;
        w = 0.0f;
        Normalize();
    }
    explicit Normal(Vector const &v)
    {
        x = v.x;
        y = v.y;
        z = v.x;
        w = 0.0f;
        Normalize();
    }

    // index access
    float &operator[](size_t index)
    {
        // return (&x)[index];
        switch (index)
        {
        case 0: return x;
        case 1: return y;
        case 2: return z;
        case 3: return w;
        default:
            return x;
        }
    }

    float operator[](size_t index) const
    {
        return const_cast<Normal *>(this)->operator[](index);
    }

    explicit operator Vector() const
    {
        return Vector{x, y, z};
    }

    void Normalize()
    {
        float fLength = std::sqrt(x * x + y * y + z * z);
        x /= fLength;
        y /= fLength;
        z /= fLength;
    }
};

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

#endif // ! RTRT_NORMAL_H
