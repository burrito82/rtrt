#ifndef RTRT_SCENE_SCENEINTERSECTLINEAR_INL
#define RTRT_SCENE_SCENEINTERSECTLINEAR_INL

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "BarycentricCoords.h"
#include "HitPoint.h"
#include "Ray.cuh"
#include "Scene.cuh"
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

__device__ __host__ __inline__
HitPoint Scene::IntersectLinear(Ray const &rRay) const
{
    HitPoint oHitPoint{};

    for (size_t iTriangleObjectIndex = 0; iTriangleObjectIndex < m_iNumberOfTriangleObjects; ++iTriangleObjectIndex)
    {
        HitPoint oTmpHitPoint = IntersectLinear(rRay, iTriangleObjectIndex, oHitPoint);
        if (oTmpHitPoint && oTmpHitPoint.m_fDistance < oHitPoint.m_fDistance)
        {
            oHitPoint = oTmpHitPoint;
        }
    }

    return oHitPoint;
}

__device__ __host__ __inline__
HitPoint Scene::IntersectLinear(Ray const &rRay, size_t iTriangleObjectIndex, HitPoint const &rHitPointBefore) const
{
    using thrust::get;
    HitPoint oHitPoint{rHitPointBefore};
    auto const &rTriangleObject = m_pTriangleObjects[iTriangleObjectIndex];
    Ray const oRay
    {
        Point{rTriangleObject.m_matInvTransformation * rRay.origin},
        Normal{rTriangleObject.m_matInvTransformation * rRay.direction}
    };
    size_t const iTriangleBegin = m_pTriangleGeometryDesc[rTriangleObject.m_iTriangleGeometry].m_iStartIndex;
    size_t const iTriangleEnd = m_pTriangleGeometryDesc[rTriangleObject.m_iTriangleGeometry].m_iNumberOfTriangles + iTriangleBegin;

    for (size_t iTriangleIndex = iTriangleBegin; iTriangleIndex < iTriangleEnd; ++iTriangleIndex)
    {
        float fDistance = IntersectTriangleWoop(oRay, GetTrianglePoints(iTriangleIndex));
        if (fDistance > 0.0f
            && fDistance < oHitPoint.m_fDistance)
        {
            oHitPoint.m_fDistance = fDistance;
            oHitPoint.m_iTriangleIndex = iTriangleIndex;
            oHitPoint.m_iObjectIndex = iTriangleObjectIndex;
        }
    }

    return oHitPoint;
}

} // namespace cuda
} // namespace rtrt

#endif // ! RTRT_SCENE_SCENEINTERSECTLINEAR_INL
