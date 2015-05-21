#ifndef RTRT_SCENE_TRIANGLE_CUH
#define RTRT_SCENE_TRIANGLE_CUH

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../math/Normal.h"
#include "../math/Point.h"

#include <thrust/tuple.h>
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

struct TriangleObjectDesc
{
    size_t m_iStartIndex;
    size_t m_iNumberOfTriangles;
};

} // namespace cuda
} // namespace rtrt

#endif // ! RTRT_SCENE_TRIANGLE_CUH

