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
    size_t m_iTriangleIndex;
    size_t m_iNumberOfTriangles;
    bool m_bIsLeaf;
    size_t m_iChildren;
};

} // namespace bvh
} // namespace rtrt

#endif // ! RTRT_SCENE_ACCEL_BVHNODE_H
