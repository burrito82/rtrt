#include "Scene.cuh"
/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../cuda/Assert.h"
#include "../cuda/Device.h"
#include "../cuda/Math.h"
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

__host__ void Raytrace(Scene const * const pScene, Ray const *pRays, size_t iNumberOfRays, HitPoint *pHitPoints)
{
    using cuda::min;
    using cuda::max;
    static int const iMaxThreads = cuda::Devices::GetInstance().Current().maxThreadsDim[0];
    static int const iMaxBlocks = cuda::Devices::GetInstance().Current().maxGridSize[0];
    dim3 blockDim;
    dim3 gridDim;
    
    for (size_t iBegin = 0ull; iBegin < iNumberOfRays;)
    {
        size_t iNumberOfRaysBatch = min(iNumberOfRays - iBegin, 81920ull);
        unsigned int iThreadsPerBlock = iMaxThreads;
        unsigned int iGridSize = min<unsigned int>((iNumberOfRaysBatch - 1u) / iThreadsPerBlock + 1u, iMaxBlocks);
        blockDim.x = iThreadsPerBlock;
        gridDim.x = iGridSize;
        Ray const *pRaysBatch = pRays + iBegin;
        HitPoint *pHitPointsBatch = pHitPoints + iBegin;
        KernelCheck();
        impl::Raytrace<<<gridDim, blockDim>>>(pScene, pRaysBatch, iNumberOfRaysBatch, pHitPointsBatch);
        iBegin += iNumberOfRaysBatch;
    };
    /*KernelCheck();
    impl::Raytrace<<<gridDim, blockDim>>>(pScene, pRays, iNumberOfRays, pHitPoints);
    KernelCheck();*/
}

namespace impl
{
__global__
void Raytrace(Scene const *pScene, Ray const *pRays, size_t iNumberOfRays, HitPoint *pHitPoints)
{
    size_t iLocalId = threadIdx.x + blockIdx.x * blockDim.x;
    while (iLocalId < iNumberOfRays)
    {
        pHitPoints[iLocalId] = pScene->Intersect(pRays[iLocalId]);
        iLocalId += gridDim.x * blockDim.x;
    }
}
} // namespace impl
} // namespace kernel

} // namespace cuda
} // namespace rtrt

