#ifndef RTRT_SCENE_DEFINES_CUH
#define RTRT_SCENE_DEFINES_CUH

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../LibraryConfig.h"
/*============================================================================*/
/* DEFINES                                                                    */
/*============================================================================*/
#define RTRTDHL __device__ __host__ __inline__
#define RTRTDH __device__ __host__
#define RTRTDHLAPI __device__ __host__ __inline__ RTRTAPI
#define RTRTDHAPI __device__ __host__ RTRTAPI

#endif // ! RTRT_SCENE_DEFINES_CUH

