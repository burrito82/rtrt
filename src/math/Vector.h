#ifndef RTRT_MATH_VECTOR_H
#define RTRT_MATH_VECTOR_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
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

struct RTRTAPI Vector : public float4
{
    Vector(float x_ = 0.0f, float y_ = 0.0f, float z_ = 0.0f)
    {
        x = x_;
        y = y_;
        z = z_;
        w = 0.0f;
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
        return const_cast<Vector *>(this)->operator[](index);
    }

    float Dot(Vector const &rhs) const
    {
        return x * rhs.x
            + y * rhs.y
            + z * rhs.z;
    }

    float AbsDot(Vector const &rhs) const
    {
        return std::abs(this->Dot(rhs));
    }

    Vector const Cross(Vector const &rhs) const
    {
        return Vector
        {
            y * rhs.z - z * rhs.y,
            z * rhs.x - x * rhs.z,
            x * rhs.y - y * rhs.x
        };
    }

    float LengthSquared() const
    {
        return this->Dot(*this);
    }

    float Length() const
    {
        return std::sqrt(this->LengthSquared());
    }

    Vector &Normalize()
    {
        float const fLength = Length();
        x /= fLength;
        y /= fLength;
        z /= fLength;
        return *this;
    }

    Vector const Normalized() const
    {
        return Vector{*this}.Normalize();
    }
};

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

#endif // ! RTRT_MATH_VECTOR_H
