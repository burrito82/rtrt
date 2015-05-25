#ifndef RTRT_SCENE_BARYCENTRICCOORDS_H
#define RTRT_SCENE_BARYCENTRICCOORDS_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Triangle.cuh"
#include "../cuda/Float4.cuh"
#include"../math/Vector.h"

#include <thrust/tuple.h>
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

/*============================================================================*/
/* CLASS DEFINITIONS                                                          */
/*============================================================================*/

struct BarycentricCoords : public Float4
{
    RTRTDH BarycentricCoords():
        Float4{}
    {

    }

    RTRTDH BarycentricCoords(TrianglePoints const &rTriangle, Point const &p)
    {
        using thrust::get;

        Vector const v0 = get<0>(rTriangle) - get<2>(rTriangle);
        Vector const v1 = get<1>(rTriangle) - get<2>(rTriangle);
        Vector const v2 = p - get<2>(rTriangle);

        float d00 = Dot(v0, v0);
        float d01 = Dot(v0, v1);
        float d10 = Dot(v1, v0);
        float d11 = Dot(v1, v1);
        float d20 = Dot(v2, v0);
        float d21 = Dot(v2, v1);

        float fInv = 1.0f / (d00 * d11 - d01 * d01);
        x = (d11 * d20 - d01 * d21) * fInv;
        y = (d00 * d21 - d01 * d20) * fInv;
        z = 1.0f - x - y;
    }

    RTRTDHL Point ToPoint(TrianglePoints const &rTriangle) const
    {
        using thrust::get;
        Point const &p0 = get<0>(rTriangle);
        Point const &p1 = get<1>(rTriangle);
        Point const &p2 = get<2>(rTriangle);

        return
        {
            x * p0.x + y * p1.x + z * p2.x,
            x * p0.y + y * p1.y + z * p2.y,
            x * p0.z + y * p1.z + z * p2.z
        };
    }

    RTRTDHL Normal ToNormal(TriangleNormals const &rTriangle) const
    {
        using thrust::get;
        Normal const &p0 = get<0>(rTriangle);
        Normal const &p1 = get<1>(rTriangle);
        Normal const &p2 = get<2>(rTriangle);

        return
        {
            x * p0.x + y * p1.x + z * p2.x,
            x * p0.y + y * p1.y + z * p2.y,
            x * p0.z + y * p1.z + z * p2.z
        };
    }
};

} // namespace cuda
} // namespace rtrt

#endif // ! RTRT_SCENE_BARYCENTRICCOORDS_H

