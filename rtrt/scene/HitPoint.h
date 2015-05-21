#ifndef RTRT_HITPOINT_H
#define RTRT_HITPOINT_H
/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
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
struct HitPoint
{
    __device__ __host__
        HitPoint():
        //m_fDistance{std::numeric_limits<float>::infinity()}
        m_fDistance{inf()}
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
};

} // namespace cuda
} // namespace rtrt

#endif // ! RTRT_HITPOINT_H

