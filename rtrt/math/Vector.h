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
    RTRTDH Vector(float x_ = 0.0f, float y_ = 0.0f, float z_ = 0.0f, float w_ = 0.0f):
        Float4{x_, y_, z_, w_}
    {
    }

    RTRTDHL explicit Vector(Float4 const &f4):
        Float4{f4}
    {

    }

    RTRTDHL float Dot(Vector const &rhs) const
    {
        return x * rhs.x
            + y * rhs.y
            + z * rhs.z;
    }

    RTRTDHL float AbsDot(Vector const &rhs) const
    {
        return std::abs(this->Dot(rhs));
    }

    RTRTDHL Vector const Cross(Vector const &rhs) const
    {
        return Vector
        {
            y * rhs.z - z * rhs.y,
            z * rhs.x - x * rhs.z,
            x * rhs.y - y * rhs.x
        };
    }

    RTRTDHL float LengthSquared() const
    {
        return this->Dot(*this);
    }

    RTRTDHL float Length() const
    {
        return std::sqrt(this->LengthSquared());
    }

    RTRTDHL Vector &Normalize()
    {
        float const fLength = Length();
        x /= fLength;
        y /= fLength;
        z /= fLength;
        return *this;
    }

    RTRTDHL Vector const Normalized() const
    {
        return Vector{*this}.Normalize();
    }
};

RTRTDHLAPI Vector const operator-(Vector const &n);
// addition
RTRTDHLAPI Vector const operator+(Vector const &lhs, Vector const &rhs);
RTRTDHLAPI Vector &operator+=(Vector &lhs, Vector const &rhs);
// subtraction
RTRTDHLAPI Vector const operator-(Vector const &lhs, Vector const &rhs);
RTRTDHLAPI Vector &operator-=(Vector &lhs, Vector const &rhs);
// scaling
RTRTDHLAPI Vector const operator*(float lhs, Vector const &rhs);
RTRTDHLAPI Vector const operator*(Vector const &lhs, float rhs);
RTRTDHLAPI Vector &operator*=(Vector &lhs, float rhs);
// inv scaling by division
RTRTDHLAPI Vector const operator/(Vector const &lhs, float rhs);
RTRTDHLAPI Vector &operator/=(Vector &lhs, float rhs);
RTRTDHLAPI float Dot(Vector const &lhs, Vector const &rhs);
RTRTDHLAPI float AbsDot(Vector const &lhs, Vector const &rhs);
RTRTDHLAPI Vector const Cross(Vector const &lhs, Vector const &rhs);
RTRTDHLAPI float LengthSquared(Vector const &v);
RTRTDHLAPI float Length(Vector const &v);
RTRTDHLAPI Vector &Normalize(Vector &v);
RTRTDHLAPI Vector const Normalized(Vector v);

/*============================================================================*/
/* OTHER HELPER FUNCTIONS                                                     */
/*============================================================================*/
__device__ __host__
void CoordinateSystem(Vector const &v0, Vector &v1, Vector &v2);

} // namespace rtrt

#include "Vector.inl"

#endif // ! RTRT_MATH_VECTOR_H
