#ifndef RTRT_SCENE_COLOR_H
#define RTRT_SCENE_COLOR_H
/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../cuda/Defines.h"
#include "../cuda/Math.h"
#include "../Align.h"
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
ALIGNED_TYPE(struct, RTRTAPI Color, 16)
{
    RTRTDH Color(float r_ = 0.0f, float g_ = 0.0f, float b_ = 0.0f, float a_ = 1.0f)
    {
        r = r_;
        g = g_;
        b = b_;
        a = a_;
    }

    RTRTDH Color &operator+=(Color const &rhs)
    {
        r += rhs.r;
        g += rhs.g;
        b += rhs.b;
        return *this;
    }

    RTRTDHL Color &operator*=(Color const &rhs)
    {
        r *= rhs.r;
        g *= rhs.g;
        b *= rhs.b;
        a *= rhs.a;
        return *this;
    }

    RTRTDHL unsigned char RedByte() const
    {
        return AsByte(r);
    }

    RTRTDHL unsigned char GreenByte() const
    {
        return AsByte(g);
    }

    RTRTDHL unsigned char BlueByte() const
    {
        return AsByte(b);
    }

    RTRTDHL unsigned char AlphaByte() const
    {
        return AsByte(a);
    }

    RTRTDHL static unsigned char AsByte(float f)
    {
        using cuda::min;
        using cuda::max;
        
        return static_cast<unsigned char>(min(255, max(0, static_cast<int>(f * 255.0f))));
    }

    float r, g, b, a;
};

RTRTDHLAPI Color const operator+(Color lhs, Color const &rhs)
{
    return lhs += rhs;
}

RTRTDHLAPI Color const operator*(Color lhs, Color const &rhs)
{
    return lhs *= rhs;
}

RTRTDHLAPI Color const operator*(float f, Color const &rhs)
{
    return
    {
        f * rhs.r,
        f * rhs.g,
        f * rhs.b,
        rhs.a,
    };
}

} // namespace rtrt

#endif // ! RTRT_SCENE_COLOR_H

