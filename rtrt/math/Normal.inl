#ifndef RTRT_MATH_NORMAL_INL
#define RTRT_MATH_NORMAL_INL
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

__device__ __host__ __inline__
Normal const operator-(Normal const &n)
{
    return Normal{-n.x, -n.y, -n.z};
}

// addition
__device__ __host__ __inline__
Normal const operator+(Normal const &lhs, Normal const &rhs)
{
    return Normal
    {
        lhs.x + rhs.x,
        lhs.y + rhs.y,
        lhs.z + rhs.z
    };
}

__device__ __host__ __inline__
Normal &operator+=(Normal &lhs, Normal const &rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    lhs.Normalize();
    return lhs;
}

// subtraction
__device__ __host__ __inline__
Normal const operator-(Normal const &lhs, Normal const &rhs)
{
    return Normal
    {
        lhs.x - rhs.x,
        lhs.y - rhs.y,
        lhs.z - rhs.z
    };
}

__device__ __host__ __inline__
Normal &operator-=(Normal &lhs, Normal const &rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    lhs.Normalize();
    return lhs;
}

// scaling
__device__ __host__ __inline__
Vector const operator*(float lhs, Normal const &rhs)
{
    return Vector
    {
        lhs * rhs.x,
        lhs * rhs.y,
        lhs * rhs.z
    };
}

__device__ __host__ __inline__
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
__device__ __host__ __inline__
Vector const operator/(Normal const &lhs, float rhs)
{
    return (1.0f / rhs) * lhs;
}

__device__ __host__ __inline__
float Dot(Normal const &lhs, Normal const &rhs)
{
    return lhs.x * rhs.x
        + lhs.y * rhs.y
        + lhs.z * rhs.z;
}

__device__ __host__ __inline__
float Dot(Vector const &lhs, Normal const &rhs)
{
    return lhs.x * rhs.x
        + lhs.y * rhs.y
        + lhs.z * rhs.z;
}

__device__ __host__ __inline__
float Dot(Normal const &lhs, Vector const &rhs)
{
    return lhs.x * rhs.x
        + lhs.y * rhs.y
        + lhs.z * rhs.z;
}

__device__ __host__ __inline__
float AbsDot(Normal const &lhs, Normal const &rhs)
{
    return std::abs(Dot(lhs, rhs));
}

__device__ __host__ __inline__
float AbsDot(Vector const &lhs, Normal const &rhs)
{
    return std::abs(Dot(lhs, rhs));
}

__device__ __host__ __inline__
float AbsDot(Normal const &lhs, Vector const &rhs)
{
    return std::abs(Dot(lhs, rhs));
}

RTRTDHLAPI Normal Reflect(Normal const &incoming, Normal const &n)
{
    return Normal{incoming - 2.0f * n * Dot(incoming, n)};
}

__device__ __host__ __inline__
Normal FaceForward(Normal const &n, Vector const &v)
{
    return (Dot(n, v) > 0.0f ? n : -n);
}

} // namespace rtrt

#endif // ! RTRT_MATH_NORMAL_INL
