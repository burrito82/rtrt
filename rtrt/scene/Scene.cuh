#ifndef RTRT_SCENE_SCENE_CUH
#define RTRT_SCENE_SCENE_CUH

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "HitPoint.h"
#include "Ray.cuh"
#include "RayTriangleIntersection.cuh"
#include "Triangle.cuh"
#include "accel/BvhBoundingBox.h"
#include "accel/BvhNode.h"
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

struct Scene
{
    __device__ __host__ __inline__
    HitPoint Intersect(Ray const &rRay) const
    {
        return IntersectBvh(rRay);
    }
    __device__ __host__
    HitPoint IntersectLinear(Ray const &rRay) const
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
    __device__ __host__
    HitPoint IntersectBvh(Ray const &rRay) const
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
    
    __device__ __host__
    TrianglePoints GetTrianglePoints(size_t iTriangleIndex) const
    {
        return
        {
            m_pPoints[3 * iTriangleIndex],
            m_pPoints[3 * iTriangleIndex + 1],
            m_pPoints[3 * iTriangleIndex + 2]
        };
    }

    __device__ __host__
    TriangleNormals GetTriangleNormals(size_t iTriangleIndex) const
    {
        return
        {
            m_pNormals[3 * iTriangleIndex],
            m_pNormals[3 * iTriangleIndex + 1],
            m_pNormals[3 * iTriangleIndex + 2]
        };
    }

    size_t m_iNumberOfTriangleObjects;
    TriangleObjectDesc *m_pTriangleObjects;
    Point *m_pPoints;
    Normal *m_pNormals;
    bvh::BvhNode *m_pBvhs;

private:
    __device__ __host__ __inline__
    HitPoint IntersectLinear(Ray const &rRay, size_t iTriangleObjectIndex, HitPoint const &rHitPointBefore) const
    {
        HitPoint oHitPoint{rHitPointBefore};
        size_t const iTriangleBegin = m_pTriangleObjects[iTriangleObjectIndex].m_iStartIndex;
        size_t const iTriangleEnd = m_pTriangleObjects[iTriangleObjectIndex].m_iNumberOfTriangles + iTriangleBegin;

        for (size_t iTriangleIndex = iTriangleBegin; iTriangleIndex < iTriangleEnd; ++iTriangleIndex)
        {
            float fDistance = IntersectTriangleWoop(rRay, GetTrianglePoints(iTriangleIndex));
            if (fDistance > 0.0f 
                && fDistance < oHitPoint.m_fDistance)
            {
                oHitPoint.m_fDistance = fDistance;
                oHitPoint.p = rRay.origin + fDistance * rRay.direction;
                oHitPoint.n = thrust::get<0>(GetTriangleNormals(iTriangleIndex)); // TODO
            }
        }

        return oHitPoint;
    }


    __device__ __host__ __inline__
    HitPoint IntersectBvh(Ray const &rRay, size_t iTriangleObjectIndex, HitPoint const &rHitPointBefore) const
    {
        HitPoint oHitPoint{rHitPointBefore};
        bvh::BvhNode const * const pRoot = &m_pBvhs[m_pTriangleObjects[iTriangleObjectIndex].m_iBvhStart];

        int aiTraversalStack[64];
        int iStackIndex = 1;
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

            if (bvh::RayBoxIntersection(rRay, pCurrentNode->m_oBoundingBox, fMinDist, fMaxDist))
            {
                if (pCurrentNode->m_bIsLeaf)
                {
                    size_t iBegin = m_pTriangleObjects[iTriangleObjectIndex].m_iStartIndex + pCurrentNode->m_iTriangleIndex;
                    size_t iEnd = iBegin + pCurrentNode->m_iNumberOfTriangles;

                    for (size_t iTriangleIndex = iBegin; iTriangleIndex < iEnd; ++iTriangleIndex)
                    {
                        float fDistance = IntersectTriangleWoop(rRay, GetTrianglePoints(iTriangleIndex));
                        if (fDistance > 0.0f && fDistance < oHitPoint.m_fDistance)
                        {
                            oHitPoint.m_fDistance = fDistance;
                            oHitPoint.p = rRay.origin + fDistance * rRay.direction;
                            oHitPoint.n = thrust::get<0>(GetTriangleNormals(iTriangleIndex)); // TODO
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
                    bool bLeftHit = bvh::RayBoxIntersection(rRay, pLeft->m_oBoundingBox, fLeftMin, fLeftMax);
                    bool bRightHit = bvh::RayBoxIntersection(rRay, pRight->m_oBoundingBox, fRightMin, fRightMax);

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
};

namespace kernel
{

__host__ void Raytrace(Scene const * const pScene, Ray const *pRays, size_t iNumberOfRays, HitPoint *pHitPoints);

namespace impl
{

__global__ void Raytrace(Scene const * const pScene, Ray const *pRays, size_t iNumberOfRays, HitPoint *pHitPoints);

} // namespace impl
} // namespace kernel

} // namespace cuda
} // namespace rtrt

#endif // ! RTRT_SCENE_SCENE_CUH

