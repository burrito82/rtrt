#ifndef RTRT_HITPOINT_H
#define RTRT_HITPOINT_H
/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "BarycentricCoords.h"

#include "../cuda/Defines.h"
#include "../math/Normal.h"
#include "../math/Point.h"

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
    RTRTDH HitPoint():
        m_fDistance{inf()},
        m_iTriangleIndex{},
        m_iObjectIndex{},
        p{},
        n{1.0f, 0.0f, 0.0f}
    {

    }

    RTRTDHL operator bool() const
    {
        return (m_fDistance > 0.0f && m_fDistance < inf());
    }
    
    RTRTDHL static float inf()
    {
        return 1.0e+30f;
    }

    float m_fDistance;
    size_t m_iTriangleIndex;
    size_t m_iObjectIndex;
    BarycentricCoords m_oBaryCoord;
    Point p;
    Normal n;
};

} // namespace cuda
} // namespace rtrt

#endif // ! RTRT_HITPOINT_H

