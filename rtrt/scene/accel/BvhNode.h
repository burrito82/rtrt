#ifndef RTRT_SCENE_ACCEL_BVHNODE_H
#define RTRT_SCENE_ACCEL_BVHNODE_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../BoundingBox.h"
/*============================================================================*/
/* DEFINES                                                                    */
/*============================================================================*/

/*============================================================================*/
/* FORWARD DECLARATIONS                                                       */
/*============================================================================*/
namespace rtrt
{
namespace bvh
{
/*============================================================================*/
/* STRUCT DEFINITIONS                                                         */
/*============================================================================*/

struct BvhNode
{
    BoundingBox m_oBoundingBox;
    bool m_bIsLeaf;
};

} // namespace bvh
} // namespace rtrt

#endif // ! RTRT_SCENE_ACCEL_BVHNODE_H