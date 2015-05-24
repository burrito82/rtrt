#ifndef RTRT_SCENE_ACCEL_BVHBOUNDINGBOX_H
#define RTRT_SCENE_ACCEL_BVHBOUNDINGBOX_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../BoundingBox.h"
#include "../Ray.cuh"
#include "../../Align.h"
#include "../../cuda/Math.h"
/*============================================================================*/
/* DEFINES                                                                    */
/*============================================================================*/

/*============================================================================*/
/* FORWARD DECLARATIONS                                                       */
/*============================================================================*/
namespace rtrt
{
namespace bvh
{
/*============================================================================*/
/* STRUCT DEFINITIONS                                                         */
/*============================================================================*/

ALIGNED_TYPE(struct, BvhBoundingBox : public BoundingBox, 16)
{
    size_t m_iTriangleIndex;
    size_t m_iNumberOfTriangles;
};

__device__ __host__ __inline__
bool RayBoxIntersection(Ray const &rRay, BoundingBox const &rBoundingBox, float &fMin, float &fMax)
{
    using cuda::min;
    using cuda::max;
    for (size_t dim = 0; dim < 3; ++dim)
    {
        if (rRay.direction[dim] == 0.0f)
        {
            if (rBoundingBox.max[dim] < rRay.origin[dim]
                || rBoundingBox.min[dim] > rRay.origin[dim])
            {
                return false;
            }
        }
        else
        {
            float inv_dir = 1.0f / rRay.direction[dim];
            float t0 = (rBoundingBox.min[dim] - rRay.origin[dim]) * inv_dir;
            float t1 = (rBoundingBox.max[dim] - rRay.origin[dim]) * inv_dir;
            fMin = max(fMin, min(t0, t1));
            fMax = min(fMax, max(t0, t1));
        }
    }

    return ((fMax >= fMin) && (fMax >= 0.0f));
}

__device__ __host__ __inline__
bool RayBoxIntersection(Ray const &rRay, BoundingBox const &rBoundingBox, 
    float &&fMax = 1.0e35f, float &&fMin = -1.0e35f)
{
    return RayBoxIntersection(rRay, rBoundingBox, fMin, fMax);
}

} // namespace bvh
} // namespace rtrt

#endif // ! RTRT_SCENE_ACCEL_BVHBOUNDINGBOX_H

