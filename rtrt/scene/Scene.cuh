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
            }
        }

        return oHitPoint;
    }


    __device__ __host__ __inline__
    HitPoint IntersectBvh(Ray const &rRay, size_t iTriangleObjectIndex, HitPoint const &rHitPointBefore) const
    {
        HitPoint oHitPoint{rHitPointBefore};
        bvh::BvhNode const * const pRoot = &m_pBvhs[m_pTriangleObjects[iTriangleObjectIndex].m_iBvhStart];

        size_t aiTraversalStack[64u]{};
        int iStackIndex = 0;
        aiTraversalStack[iStackIndex] = 0u;

        do
        {
            size_t const iCurrentNodeIndex = aiTraversalStack[iStackIndex--];
            bvh::BvhNode const &rCurrentNode = pRoot[iCurrentNodeIndex];
            if (bvh::RayBoxIntersection(rRay, rCurrentNode.m_oBoundingBox, oHitPoint.m_fDistance))
            {
                if (rCurrentNode.m_bIsLeaf)
                {
                    size_t iBegin = m_pTriangleObjects[iTriangleObjectIndex].m_iStartIndex + rCurrentNode.m_iTriangleIndex;
                    size_t iEnd = iBegin + rCurrentNode.m_iNumberOfTriangles;
                    for (size_t iTriangleIndex = iBegin; iTriangleIndex < iEnd; ++iTriangleIndex)
                    {
                        float fDistance = IntersectTriangleWoop(rRay, GetTrianglePoints(iTriangleIndex));
                        if (fDistance > 0.0f && fDistance < oHitPoint.m_fDistance)
                        {
                            oHitPoint.m_fDistance = fDistance;
                        }
                    }
                }
                else
                {
                    aiTraversalStack[++iStackIndex] = 2u * iCurrentNodeIndex + 1u;
                    aiTraversalStack[++iStackIndex] = 2u * iCurrentNodeIndex + 2u;
                }
            }
        }
        while (iStackIndex >= 0);

        return oHitPoint;
    }
};

namespace kernel
{

__host__ void Raytrace(dim3 blockDim, dim3 gridDim, Scene const * const pScene, Ray const *pRays, size_t iNumberOfRays, HitPoint *pHitPoints);

namespace impl
{

__global__ void Raytrace(Scene const * const pScene, Ray const *pRays, size_t iNumberOfRays, HitPoint *pHitPoints);

} // namespace impl
} // namespace kernel

} // namespace cuda
} // namespace rtrt

#endif // ! RTRT_SCENE_SCENE_CUH

