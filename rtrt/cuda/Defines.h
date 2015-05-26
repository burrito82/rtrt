#ifndef RTRT_SCENE_DEFINES_CUH
#define RTRT_SCENE_DEFINES_CUH

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../LibraryConfig.h"
/*============================================================================*/
/* DEFINES                                                                    */
/*============================================================================*/

#ifdef RTRT_USE_CUDA
#define RTRTDHL __device__ __host__ __inline__
#define RTRTDH __device__ __host__
#define RTRTDHLAPI __device__ __host__ __inline__ RTRTAPI
#define RTRTDHAPI __device__ __host__ RTRTAPI
#else // ! RTRT_USE_CUDA
#define __host__
#define __device__
#define __inline__ inline
#define RTRTDHL inline
#define RTRTDH 
#define RTRTDHLAPI inline RTRTAPI
#define RTRTDHAPI RTRTAPI
#endif // ! RTRT_USE_CUDA

#endif // ! RTRT_SCENE_DEFINES_CUH

