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

    explicit Vector(Float4 const &f4):
        Float4{f4}
    {

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

RTRTAPI Vector const operator-(Vector const &n);
// addition
RTRTAPI Vector const operator+(Vector const &lhs, Vector const &rhs);
RTRTAPI Vector &operator+=(Vector &lhs, Vector const &rhs);
// subtraction
RTRTAPI Vector const operator-(Vector const &lhs, Vector const &rhs);
RTRTAPI Vector &operator-=(Vector &lhs, Vector const &rhs);
// scaling
RTRTAPI Vector const operator*(float lhs, Vector const &rhs);
RTRTAPI Vector const operator*(Vector const &lhs, float rhs);
RTRTAPI Vector &operator*=(Vector &lhs, float rhs);
// inv scaling by division
RTRTAPI Vector const operator/(Vector const &lhs, float rhs);
RTRTAPI Vector &operator/=(Vector &lhs, float rhs);
RTRTAPI float Dot(Vector const &lhs, Vector const &rhs);
RTRTAPI float AbsDot(Vector const &lhs, Vector const &rhs);
RTRTAPI Vector const Cross(Vector const &lhs, Vector const &rhs);
RTRTAPI float LengthSquared(Vector const &v);
RTRTAPI float Length(Vector const &v);
RTRTAPI Vector &Normalize(Vector &v);
RTRTAPI Vector const Normalized(Vector v);
/*============================================================================*/
/* OTHER HELPER FUNCTIONS                                                     */
/*============================================================================*/

void CoordinateSystem(Vector const &v0, Vector &v1, Vector &v2);

} // namespace rtrt

#endif // ! RTRT_MATH_VECTOR_H
