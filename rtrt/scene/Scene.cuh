#ifndef RTRT_SCENE_SCENE_CUH
#define RTRT_SCENE_SCENE_CUH

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "HitPoint.h"
#include "Material.h"
#include "PointLight.h"
#include "Ray.cuh"
#include "RayTriangleIntersection.cuh"
#include "Triangle.h"
#include "TriangleGeometryDesc.h"
#include "TriangleObject.h"
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
        using thrust::get;

        auto oHitPoint = IntersectBvh(rRay);
        if (oHitPoint)
        {
            auto const oTrianglePoints = GetTrianglePoints(oHitPoint.m_iTriangleIndex);
            auto const oTriangleNormals = GetTriangleNormals(oHitPoint.m_iTriangleIndex);
            auto const &rTriangleObject = m_pTriangleObjects[oHitPoint.m_iObjectIndex];
            oHitPoint.p = rRay.origin + oHitPoint.m_fDistance * rRay.direction;
            oHitPoint.m_oBaryCoord = BarycentricCoords{oTrianglePoints, Point{rTriangleObject.m_matInvTransformation * oHitPoint.p}};
            oHitPoint.n = Normal{rTriangleObject.m_matTransformation * oHitPoint.m_oBaryCoord.ToNormal(oTriangleNormals)};
        }
        return oHitPoint;
    }
    
    RTRTDHL TrianglePoints GetTrianglePoints(std::size_t iTriangleIndex) const
    {
        return
        {
            m_pPoints[3 * iTriangleIndex],
            m_pPoints[3 * iTriangleIndex + 1],
            m_pPoints[3 * iTriangleIndex + 2]
        };
    }

    RTRTDHL TriangleNormals GetTriangleNormals(std::size_t iTriangleIndex) const
    {
        return
        {
            m_pNormals[3 * iTriangleIndex],
            m_pNormals[3 * iTriangleIndex + 1],
            m_pNormals[3 * iTriangleIndex + 2]
        };
    }

    std::size_t m_iNumberOfTriangleObjects;
    TriangleGeometryDesc *m_pTriangleGeometryDesc;
    Point *m_pPoints;
    Normal *m_pNormals;
    bvh::BvhNode *m_pBvhs;

    TriangleObject *m_pTriangleObjects;
    Material *m_pMaterials;
    PointLight *m_pPointLights;

private:
    RTRTDHL HitPoint IntersectLinear(Ray const &rRay) const;
    RTRTDHL HitPoint IntersectLinear(Ray const &rRay, std::size_t iTriangleObjectIndex, HitPoint const &rHitPointBefore) const;

    RTRTDHL HitPoint IntersectBvh(Ray const &rRay) const;
    RTRTDHL HitPoint IntersectBvh(Ray const &rRay, std::size_t iTriangleObjectIndex, HitPoint const &rHitPointBefore) const;
};

namespace kernel
{

__host__ void Raytrace(Scene const * const pScene, Ray const *pRays, std::size_t iNumberOfRays, HitPoint *pHitPoints);

namespace impl
{

__global__ void Raytrace(Scene const * const pScene, Ray const *pRays, std::size_t iNumberOfRays, HitPoint *pHitPoints);

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

