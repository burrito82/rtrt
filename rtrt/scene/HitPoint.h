#ifndef RTRT_HITPOINT_H
#define RTRT_HITPOINT_H
/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../math/Normal.h"
#include "../math/Point.h"
#include "../Align.h"
#include <limits>
#include <cfloat>
/*============================================================================*/
/* DEFINES                                                                    */
/*============================================================================*/

/*============================================================================*/
/* FORWARD DECLARATIONS                                                       */
/*============================================================================*/
namespace rtrt
{
namespace cuda
{
/*============================================================================*/
/* STRUCT DEFINITIONS                                                         */
/*============================================================================*/
//ALIGNED_TYPE(struct, HitPoint, 64)
struct HitPoint
{
    __device__ __host__
    HitPoint():
        m_fDistance{inf()},
        p{},
        n{1.0f, 0.0f, 0.0f}
    {

    }

    __device__ __host__
    operator bool() const
    {
        return (m_fDistance > 0.0f && m_fDistance < inf());
    }

    __device__ __host__
    static float inf()
    {
        return 1.0e+30f;
    }

    float m_fDistance;
    Point p;
    Normal n;
};

} // namespace cuda
} // namespace rtrt

#endif // ! RTRT_HITPOINT_H

