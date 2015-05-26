#ifndef RTRT_SCENE_ACCEL_SCENEINTERSECTBVH_INL
#define RTRT_SCENE_ACCEL_SCENEINTERSECTBVH_INL

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../HitPoint.h"
#include "../Ray.cuh"
#include "../RayTriangleIntersection.cuh"
#include "../Triangle.h"
#include "BvhBoundingBox.h"
#include "BvhNode.h"
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
HitPoint Scene::IntersectBvh(Ray const &rRay) const
{
    HitPoint oHitPoint{};

    for (size_t iTriangleObjectIndex = 0; iTriangleObjectIndex < m_iNumberOfTriangleObjects; ++iTriangleObjectIndex)
    {
        HitPoint oTmpHitPoint = IntersectBvh(rRay, iTriangleObjectIndex, oHitPoint);
        if (oTmpHitPoint && oTmpHitPoint.m_fDistance < oHitPoint.m_fDistance)
        {
            oHitPoint = oTmpHitPoint;
        }
    }

    return oHitPoint;
}

__device__ __host__ __inline__
HitPoint Scene::IntersectBvh(Ray const &rRay, size_t iTriangleObjectIndex, HitPoint const &rHitPointBefore) const
{
    HitPoint oHitPoint{rHitPointBefore};
    auto const &rTriangleObject = m_pTriangleObjects[iTriangleObjectIndex];
    Ray const oRay
    {
        Point{rTriangleObject.m_matInvTransformation * rRay.origin},
        Normal{rTriangleObject.m_matInvTransformation * rRay.direction}
    };
    auto iGeometryIndex = m_pTriangleObjects[rTriangleObject.m_iTriangleGeometry].m_iTriangleGeometry;
    bvh::BvhNode const * const pRoot = &m_pBvhs[m_pTriangleGeometryDesc[iGeometryIndex].m_iBvhStart];

    int aiTraversalStack[64];
    int iStackIndex = 0;
    aiTraversalStack[0] = -1;
    int iCurrentNodeIndex = 0;
    bvh::BvhNode const *pCurrentNode = nullptr;
    float fMinDist;
    float fMaxDist;

    do
    {
        pCurrentNode = &pRoot[iCurrentNodeIndex];
        fMinDist = -1.0e35f;
        fMaxDist = oHitPoint.m_fDistance;

        if (bvh::RayBoxIntersection(oRay, pCurrentNode->m_oBoundingBox, fMinDist, fMaxDist))
        {
            if (pCurrentNode->m_bIsLeaf)
            {
                size_t iBegin = m_pTriangleGeometryDesc[iGeometryIndex].m_iStartIndex + pCurrentNode->m_iTriangleIndex;
                size_t iEnd = iBegin + pCurrentNode->m_iNumberOfTriangles;

                for (size_t iTriangleIndex = iBegin; iTriangleIndex < iEnd; ++iTriangleIndex)
                {
                    float fDistance = IntersectTriangleWoop(oRay, GetTrianglePoints(iTriangleIndex));
                    if (fDistance > 0.0f && fDistance < oHitPoint.m_fDistance)
                    {
                        oHitPoint.m_fDistance = fDistance;
                        oHitPoint.m_iTriangleIndex = iTriangleIndex;
                        oHitPoint.m_iObjectIndex = iTriangleObjectIndex;
                    }
                }

                iCurrentNodeIndex = aiTraversalStack[iStackIndex--];
            }
            else
            {
                int iLeft = 2 * iCurrentNodeIndex + 1;
                int iRight = 2 * iCurrentNodeIndex + 2;
                bvh::BvhNode const *pLeft = &pRoot[iLeft];
                bvh::BvhNode const *pRight = &pRoot[iRight];
                float fLeftMin = -1.0e35f, fLeftMax = oHitPoint.m_fDistance;
                float fRightMin = -1.0e35f, fRightMax = oHitPoint.m_fDistance;
                bool bLeftHit = bvh::RayBoxIntersection(oRay, pLeft->m_oBoundingBox, fLeftMin, fLeftMax);
                bool bRightHit = bvh::RayBoxIntersection(oRay, pRight->m_oBoundingBox, fRightMin, fRightMax);

                if (bLeftHit && bRightHit)
                {
                    if (fLeftMin < fRightMin)
                    {
                        iCurrentNodeIndex = iLeft;
                        aiTraversalStack[++iStackIndex] = iRight;
                    }
                    else
                    {
                        iCurrentNodeIndex = iRight;
                        aiTraversalStack[++iStackIndex] = iLeft;
                    }
                }
                else
                {
                    if (bLeftHit)
                    {
                        iCurrentNodeIndex = iLeft;
                    }
                    else
                    {
                        if (bRightHit)
                        {
                            iCurrentNodeIndex = iRight;
                        }
                        else
                        {
                            iCurrentNodeIndex = aiTraversalStack[iStackIndex--];
                        }
                    }
                }
            }
        }
        else
        {
            iCurrentNodeIndex = aiTraversalStack[iStackIndex--];
        }
    }
    while (iCurrentNodeIndex >= 0);

    return oHitPoint;
}

#ifdef RTRT_USE_CUDA
namespace kernel
{

__host__ void Raytrace(Scene const * const pScene, Ray const *pRays, size_t iNumberOfRays, HitPoint *pHitPoints);

namespace impl
{

__global__ void Raytrace(Scene const * const pScene, Ray const *pRays, size_t iNumberOfRays, HitPoint *pHitPoints);

} // namespace impl
} // namespace kernel
#endif // ! RTRT_USE_CUDA

} // namespace cuda
} // namespace rtrt

#endif // ! RTRT_SCENE_ACCEL_SCENEINTERSECTBVH_INL

