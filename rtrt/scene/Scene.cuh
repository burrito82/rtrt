#ifndef RTRT_SCENE_SCENE_CUH
#define RTRT_SCENE_SCENE_CUH

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "HitPoint.h"
#include "Ray.cuh"
#include "RayTriangleIntersection.cuh"
#include "Triangle.cuh"
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
    __device__ __host__
    HitPoint Intersect(Ray const &rRay) const
    {
        HitPoint oHitPoint{};

        for (size_t iTriangleObjectIndex = 0; iTriangleObjectIndex < m_iNumberOfTriangleObjects; ++iTriangleObjectIndex)
        {
            size_t const iTriangleBegin = m_pTriangleObjects[iTriangleObjectIndex].m_iStartIndex;
            size_t const iTriangleEnd = m_pTriangleObjects[iTriangleObjectIndex].m_iNumberOfTriangles + iTriangleBegin;

            for (size_t iTriangleIndex = iTriangleBegin; iTriangleIndex < iTriangleEnd; ++iTriangleIndex)
            {
                float fDistance = IntersectTriangleWoop(rRay, GetTrianglePoints(iTriangleIndex));
                if (fDistance > 0.0f && fDistance < oHitPoint.m_fDistance)
                {
                    oHitPoint.m_fDistance = fDistance;
                }
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

