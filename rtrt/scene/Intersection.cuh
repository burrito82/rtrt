#ifndef RTRT_SCENE_INTERSECTION_CUH
#define RTRT_SCENE_INTERSECTION_CUH

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
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
__device__ __host__
void Intersect(Scene const *pScene, Ray const *pRay, HitPoint *pHitPoints)
{
    //auto iLocalId = threadIdx.x + blockIdx.x * gridDim.x;

    //pHitPoints[iLocalId]
}

} // namespace cuda
} // namespace rtrt

#endif // ! RTRT_SCENE_INTERSECTION_CUH
