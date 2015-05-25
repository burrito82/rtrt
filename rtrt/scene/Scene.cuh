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
#include "../cuda/Defines.h"
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
    RTRTDHL HitPoint Intersect(Ray const &rRay) const
    {
        return IntersectBvh(rRay);
    }

    RTRTDHL HitPoint IntersectLinear(Ray const &rRay) const;

    RTRTDHL HitPoint IntersectBvh(Ray const &rRay) const;
    
    RTRTDHL TrianglePoints GetTrianglePoints(size_t iTriangleIndex) const
    {
        return
        {
            m_pPoints[3 * iTriangleIndex],
            m_pPoints[3 * iTriangleIndex + 1],
            m_pPoints[3 * iTriangleIndex + 2]
        };
    }

    RTRTDHL TriangleNormals GetTriangleNormals(size_t iTriangleIndex) const
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
    RTRTDHL HitPoint IntersectLinear(Ray const &rRay, size_t iTriangleObjectIndex, HitPoint const &rHitPointBefore) const;

    RTRTDHL HitPoint IntersectBvh(Ray const &rRay, size_t iTriangleObjectIndex, HitPoint const &rHitPointBefore) const;
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

/*============================================================================*/
/* METHOD IMPLEMENTATIONS                                                     */
/*============================================================================*/

#include "SceneIntersectLinear.inl"
#include "accel/SceneIntersectBvh.inl"

#endif // ! RTRT_SCENE_SCENE_CUH

