#ifndef RTRT_CUDA_MATH_CUH
#define RTRT_CUDA_MATH_CUH

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#ifdef RTRT_USE_CUDA
#include <cuda_runtime.h>
#endif // ! RTRT_USE_CUDA
/*============================================================================*/
/* DEFINES                                                                    */
/*============================================================================*/
#ifndef RTRT_USE_CUDA
#define __device__
#define __host__
#define __inline__ inline
#endif // ! RTRT_USE_CUDA
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

template<typename T>
__device__ __host__ __inline__
T min(T lhs, T rhs)
{
    return (lhs < rhs) ? lhs : rhs;
}

template<typename T>
__device__ __host__ __inline__
T max(T lhs, T rhs)
{
    return (lhs > rhs) ? lhs : rhs;
}

} // namespace cuda
} // namespace rtrt

#endif // ! RTRT_CUDA_MATH_CUH
