#ifndef RTRT_SCENE_INTERSECTION_H
#define RTRT_SCENE_INTERSECTION_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Ray.h"
#include "../math/Normal.h"
#include "../math/Point.h"
#include "../LibraryConfig.h"

#include <limits>
/*============================================================================*/
/* DEFINES                                                                    */
/*============================================================================*/

/*============================================================================*/
/* FORWARD DECLARATIONS                                                       */
/*============================================================================*/
namespace rtrt
{
/*============================================================================*/
/* STRUCT DEFINITIONS                                                         */
/*============================================================================*/

struct RTRTAPI Intersection
{
    float distance;
    Point hitpoint;
    Normal surface_normal;

    Intersection():
        distance{std::numeric_limits<float>::max()}
    {
    }

    Intersection(Ray const &ray, float distance_):
        distance{distance_},
        hitpoint{ray.origin + distance * ray.direction}
    {
    }
};

} // namespace rtrt

#endif // ! RTRT_SCENE_INTERSECTION_H
