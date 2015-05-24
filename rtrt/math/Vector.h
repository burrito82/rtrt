#ifndef RTRT_MATH_VECTOR_H
#define RTRT_MATH_VECTOR_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../LibraryConfig.h"

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

struct RTRTAPI Vector : public Float4
{
    __device__ __host__
    Vector(float x_ = 0.0f, float y_ = 0.0f, float z_ = 0.0f, float w_ = 0.0f):
        Float4{x_, y_, z_, w_}
    {
    }

    __device__ __host__
    explicit Vector(Float4 const &f4):
        Float4{f4}
    {

    }

    __device__ __host__
    float Dot(Vector const &rhs) const
    {
        return x * rhs.x
            + y * rhs.y
            + z * rhs.z;
    }

    __device__ __host__
    float AbsDot(Vector const &rhs) const
    {
        return std::abs(this->Dot(rhs));
    }

    __device__ __host__
    Vector const Cross(Vector const &rhs) const
    {
        return Vector
        {
            y * rhs.z - z * rhs.y,
            z * rhs.x - x * rhs.z,
            x * rhs.y - y * rhs.x
        };
    }

    __device__ __host__
    float LengthSquared() const
    {
        return this->Dot(*this);
    }

    __device__ __host__
    float Length() const
    {
        return std::sqrt(this->LengthSquared());
    }

    __device__ __host__
    Vector &Normalize()
    {
        float const fLength = Length();
        x /= fLength;
        y /= fLength;
        z /= fLength;
        return *this;
    }

    __device__ __host__
    Vector const Normalized() const
    {
        return Vector{*this}.Normalize();
    }
};

#define RMVHL __device__ __host__ __inline__ RTRTAPI

RMVHL Vector const operator-(Vector const &n);
// addition
RMVHL Vector const operator+(Vector const &lhs, Vector const &rhs);
RMVHL Vector &operator+=(Vector &lhs, Vector const &rhs);
// subtraction
RMVHL Vector const operator-(Vector const &lhs, Vector const &rhs);
RMVHL Vector &operator-=(Vector &lhs, Vector const &rhs);
// scaling
RMVHL Vector const operator*(float lhs, Vector const &rhs);
RMVHL Vector const operator*(Vector const &lhs, float rhs);
RMVHL Vector &operator*=(Vector &lhs, float rhs);
// inv scaling by division
RMVHL Vector const operator/(Vector const &lhs, float rhs);
RMVHL Vector &operator/=(Vector &lhs, float rhs);
RMVHL float Dot(Vector const &lhs, Vector const &rhs);
RMVHL float AbsDot(Vector const &lhs, Vector const &rhs);
RMVHL Vector const Cross(Vector const &lhs, Vector const &rhs);
RMVHL float LengthSquared(Vector const &v);
RMVHL float Length(Vector const &v);
RMVHL Vector &Normalize(Vector &v);
RMVHL Vector const Normalized(Vector v);

#undef RMVHL

/*============================================================================*/
/* OTHER HELPER FUNCTIONS                                                     */
/*============================================================================*/
__device__ __host__
void CoordinateSystem(Vector const &v0, Vector &v1, Vector &v2);

} // namespace rtrt

#include "Vector.inl"

#endif // ! RTRT_MATH_VECTOR_H
