#include "Scene.cuh"
/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../cuda/Assert.h"
/*============================================================================*/
/* MACROS AND DEFINES, CONSTANTS AND STATICS                                  */
/*============================================================================*/
namespace rtrt
{
namespace cuda
{
/*============================================================================*/
/* LOCAL VARS AND FUNCS                                                       */
/*============================================================================*/

/*============================================================================*/
/* CONSTRUCTORS / DESTRUCTOR                                                  */
/*============================================================================*/

/*============================================================================*/
/* IMPLEMENTATION                                                             */
/*============================================================================*/

namespace kernel
{

__host__ void Raytrace(dim3 blockDim, dim3 gridDim, Scene const * const pScene, Ray const *pRays, size_t iNumberOfRays, HitPoint *pHitPoints)
{
    KernelCheck();
    impl::Raytrace<<<gridDim, blockDim>>>(pScene, pRays, iNumberOfRays, pHitPoints);
    KernelCheck();
}

namespace impl
{
__global__
void Raytrace(Scene const *pScene, Ray const *pRays, size_t iNumberOfRays, HitPoint *pHitPoints)
{
    int iLocalId = threadIdx.x + blockIdx.x * blockDim.x;
    if (iLocalId < iNumberOfRays)
    {
        pHitPoints[iLocalId] = pScene->Intersect(pRays[iLocalId]);
    }
}
} // namespace impl
} // namespace kernel

} // namespace cuda
} // namespace rtrt

