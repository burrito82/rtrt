#ifndef RTRT_CUDA_TYPEVERIFIER_CU
#define RTRT_CUDA_TYPEVERIFIER_CU
#include "TypeVerifier.cuh"
/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Assert.h"
#include "VectorMemory.h"
#include "Float4.cuh"
#include "../math/Normal.h"
#include "../math/Point.h"
#include "../math/Vector.h"
#include "../scene/HitPoint.h"
#include "../scene/Material.h"
#include "../scene/Ray.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
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
namespace kernel
{

template<typename T>
__global__
void GetTypeSize(size_t *piSize)
{
    *piSize = sizeof(T);
}

} // namespace kernel

template<typename T>
__host__
size_t GetTypeSize()
{
    VectorMemory<size_t, GPU_TO_CPU> vecSizes(1);
    KernelCheck();
    (kernel::GetTypeSize<T>)<<<1, 1>>>(vecSizes.CudaPointer());
    KernelCheck();
    vecSizes.Synchronize();
    return vecSizes[0];
}

__host__
size_t GetTypeSize(HitPoint const &)
{
    return GetTypeSize<HitPoint>();
}

__host__
size_t GetTypeSize(Ray const &)
{
    return GetTypeSize<Ray>();
}

__host__
size_t GetTypeSize(Float4 const &)
{
    return GetTypeSize<Float4>();
}

__host__
size_t GetTypeSize(Material const &)
{
    return GetTypeSize<Material>();
}

__host__
size_t GetTypeSize(Normal const &)
{
    return GetTypeSize<Normal>();
}

__host__
size_t GetTypeSize(Point const &)
{
    return GetTypeSize<Point>();
}

__host__
size_t GetTypeSize(Vector const &)
{
    return GetTypeSize<Vector>();
}

} // namespace cuda
} // namespace rtrt


#endif // ! RTRT_CUDA_TYPEVERIFIER_CU

