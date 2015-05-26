#ifndef RTRT_MATH_Float4_H
#define RTRT_MATH_Float4_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Defines.h"

#ifdef RTRT_USE_CUDA
#include <vector_functions.h>
#include <vector_types.h>
#endif // ! RTRT_USE_CUDA
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

#ifndef RTRT_USE_CUDA
struct float4
{
	float x, y, z, w;
};

static float4 make_float4(float x = 0.0f, float y = 0.0f, float z = 0.0f, float w = 0.0f)
{
	return float4{x, y, z, w};
}
#endif // ! RTRT_USE_CUDA

struct Float4 : public float4
{
    RTRTDH Float4(float x_ = 0.0f, float y_ = 0.0f, float z_ = 0.0f, float w_ = 0.0f)
    {
        x = x_;
        y = y_;
        z = z_;
        w = w_;
    }

    // index access
    RTRTDHL float &operator[](size_t index)
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

    RTRTDHL float operator[](size_t index) const
    {
        return const_cast<Float4 *>(this)->operator[](index);
    }

    RTRTDHL Float4 const operator-() const
    {
        return{-x, -y, -z, -w};
    }

    // addition
    RTRTDHL Float4 const operator+(Float4 const &rhs) const
    {
        return {x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w};
    }

    __device__ __host__
    Float4 &operator+=(Float4 const &rhs)
    {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        w += rhs.w;
        return *this;
    }

    // subtraction
    RTRTDHL Float4 const operator-(Float4 const &rhs) const
    {
        return{x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w};
    }

    RTRTDHL Float4 &operator-=(Float4 const &rhs)
    {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        w -= rhs.w;
        return *this;
    }

    // scaling

    RTRTDHL Float4 const operator*(float f) const
    {
        return{x * f , y * f, z * f, w * f};
    }

    RTRTDHL Float4 &operator*=(float rhs)
    {
        x *= rhs;
        y *= rhs;
        z *= rhs;
        w *= rhs;
        return *this;
    }

    // inv scaling by division
    RTRTDHL Float4 const operator/(float rhs) const
    {
        float fInv = 1.0f / rhs;
        return{x * fInv, y * fInv, z * fInv, w * fInv};
    }

    RTRTDHL Float4 &operator/=(float rhs)
    {
        float fInv = 1.0f / rhs;
        *this *= fInv;
        return *this;
    }
};

template<typename F4>
RTRTDHL F4 &Denormalize(F4 &f4)
{
    f4 /= f4.w;
    return f4;
}

template<typename F4>
RTRTDHL F4 Denormalized(F4 f4)
{
    return f4 /= f4.w;
}

RTRTDHL float4 const operator-(float4 const &f4)
{
    return make_float4(-f4.x, -f4.y, -f4.z, -f4.w);
}

// addition
RTRTDHL float4 const operator+(float4 const &lhs, float4 const &rhs)
{
    return make_float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}

RTRTDHL float4 &operator+=(float4 &lhs, float4 const &rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    lhs.w += rhs.w;
    return lhs;
}

// subtraction
RTRTDHL float4 const operator-(float4 const &lhs, float4 const &rhs)
{
    return make_float4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
}

RTRTDHL float4 &operator-=(float4 &lhs, float4 const &rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    lhs.w -= rhs.w;
    return lhs;
}

// scaling
RTRTDHL float4 const operator*(float lhs, float4 const &rhs)
{
    return make_float4(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w);
}

RTRTDHL float4 const operator*(float4 const &lhs, float f)
{
    return make_float4(lhs.x * f, lhs.y * f, lhs.z * f, lhs.w * f);
}

RTRTDHL float4 &operator*=(float4 &lhs, float rhs)
{
    lhs.x *= rhs;
    lhs.y *= rhs;
    lhs.z *= rhs;
    lhs.w *= rhs;
    return lhs;
}

// inv scaling by division
RTRTDHL float4 const operator/(float4 const &lhs, float rhs)
{
    float fInv = 1.0f / rhs;
    return make_float4(lhs.x * fInv, lhs.y * fInv, lhs.z * fInv, lhs.w * fInv);
}

RTRTDHL float4 &operator/=(float4 &lhs, float rhs)
{
    float fInv = 1.0f / rhs;
    lhs *= fInv;
    return lhs;
}

} // namespace rtrt

#endif // ! RTRT_MATH_Float4_H
