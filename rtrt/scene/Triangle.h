#ifndef RTRT_SCENE_TRIANGLE_H
#define RTRT_SCENE_TRIANGLE_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../math/Normal.h"
#include "../math/Point.h"

#ifdef RTRT_USE_CUDA
#include <thrust/tuple.h>
#else // ! RTRT_USE_CUDA
#include <tuple>
namespace thrust = std;
#endif // ! RTRT_USE_CUDA
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

/*============================================================================*/
/* CLASS DEFINITIONS                                                          */
/*============================================================================*/

using TrianglePoints = thrust::tuple<Point, Point, Point>;
using TriangleNormals = thrust::tuple<Normal, Normal, Normal>;

} // namespace cuda
} // namespace rtrt

#endif // ! RTRT_SCENE_TRIANGLE_H

