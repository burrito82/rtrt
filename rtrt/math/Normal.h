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
    RTRTDH Normal()
    {
        x = y = z = w = -1.0f;
    }
    RTRTDH Normal(float x_, float y_, float z_):
        Float4{x_, y_, z_, 0.0f}
    {
        Normalize();
    }
    RTRTDH explicit Normal(Vector const &v):
        Float4{v.x, v.y, v.z, 0.0f}
    {
        Normalize();
    }
    RTRTDH explicit Normal(Float4 const &f4):
        Float4{f4}
    {

    }

    RTRTDHL explicit operator Vector() const
    {
        return Vector{x, y, z};
    }

    RTRTDHL void Normalize()
    {
        float fLength = std::sqrt(x * x + y * y + z * z);
        x /= fLength;
        y /= fLength;
        z /= fLength;
    }
};

RTRTDHLAPI Normal const operator-(Normal const &n);

// addition
RTRTDHLAPI Normal const operator+(Normal const &lhs, Normal const &rhs);
RTRTDHLAPI Normal &operator+=(Normal &lhs, Normal const &rhs);

// subtraction
RTRTDHLAPI Normal const operator-(Normal const &lhs, Normal const &rhs);
RTRTDHLAPI Normal &operator-=(Normal &lhs, Normal const &rhs);

// scaling
RTRTDHLAPI Vector const operator*(float lhs, Normal const &rhs);
RTRTDHLAPI Vector const operator*(Normal const &lhs, float rhs);

// inv scaling by division
RTRTDHLAPI Vector const operator/(Normal const &lhs, float rhs);
RTRTDHLAPI float Dot(Normal const &lhs, Normal const &rhs);
RTRTDHLAPI float Dot(Vector const &lhs, Normal const &rhs);
RTRTDHLAPI float Dot(Normal const &lhs, Vector const &rhs);
RTRTDHLAPI float AbsDot(Normal const &lhs, Normal const &rhs);
RTRTDHLAPI float AbsDot(Vector const &lhs, Normal const &rhs);
RTRTDHLAPI float AbsDot(Normal const &lhs, Vector const &rhs);
RTRTDHLAPI Normal FaceForward(Normal const &n, Vector const &v);

} // namespace rtrt

#include "Normal.inl"

#endif // ! RTRT_MATH_NORMAL_H
