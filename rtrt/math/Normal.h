#ifndef RTRT_MATH_NORMAL_H
#define RTRT_MATH_NORMAL_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Vector.h"
#include "../cuda/Float4.cuh"
#include "../LibraryConfig.h"
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

struct RTRTAPI Normal : public Float4
{
    __device__ __host__
    Normal()
    {
        x = y = z = w = -1.0f;
    }
    __device__ __host__
    Normal(float x_, float y_, float z_):
        Float4{x_, y_, z_, 0.0f}
    {
        Normalize();
    }
    __device__ __host__
    explicit Normal(Vector const &v):
        Float4{v.x, v.y, v.z, 0.0f}
    {
        Normalize();
    }
    __device__ __host__
    explicit Normal(Float4 const &f4):
        Float4{f4}
    {

    }

    __device__ __host__
    explicit operator Vector() const
    {
        return Vector{x, y, z};
    }

    __device__ __host__
    void Normalize()
    {
        float fLength = std::sqrt(x * x + y * y + z * z);
        x /= fLength;
        y /= fLength;
        z /= fLength;
    }
};

RTRTAPI Normal const operator-(Normal const &n);

// addition
RTRTAPI Normal const operator+(Normal const &lhs, Normal const &rhs);
RTRTAPI Normal &operator+=(Normal &lhs, Normal const &rhs);

// subtraction
RTRTAPI Normal const operator-(Normal const &lhs, Normal const &rhs);
RTRTAPI Normal &operator-=(Normal &lhs, Normal const &rhs);

// scaling
RTRTAPI Vector const operator*(float lhs, Normal const &rhs);
RTRTAPI Vector const operator*(Normal const &lhs, float rhs);

// inv scaling by division
RTRTAPI Vector const operator/(Normal const &lhs, float rhs);
RTRTAPI float Dot(Normal const &lhs, Normal const &rhs);
RTRTAPI float Dot(Vector const &lhs, Normal const &rhs);
RTRTAPI float Dot(Normal const &lhs, Vector const &rhs);
RTRTAPI float AbsDot(Normal const &lhs, Normal const &rhs);
RTRTAPI float AbsDot(Vector const &lhs, Normal const &rhs);
RTRTAPI float AbsDot(Normal const &lhs, Vector const &rhs);
RTRTAPI Normal FaceForward(Normal const &n, Vector const &v);

} // namespace rtrt

#include "Normal.inl"

#endif // ! RTRT_MATH_NORMAL_H
