#ifndef RTRT_SCENE_ACCEL_BVHCONSTRUCTOR_H
#define RTRT_SCENE_ACCEL_BVHCONSTRUCTOR_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "BvhNode.h"

#include <memory>
#include <vector>
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
//class BvhTmpNode;
/*============================================================================*/
/* STRUCT DEFINITIONS                                                         */
/*============================================================================*/

/*============================================================================*/
/* CLASS DEFINITIONS                                                          */
/*============================================================================*/

/**
 * @param
 * @return
 * @see
 * @todo
 * @bug
 * @deprecated
 */
class BvhBuilder
{
public:
    //BvhBuilder(cuda::TriangleObjectDesc const &oTriangleObjDesc);

    std::vector<BvhNode> GetBvh()
    { }
protected:
private:


    //std::shared_ptr<BvhTmpNode> m_pRoot;
};

} // namespace bvh
} // namespace rtrt

#endif // ! RTRT_SCENE_ACCEL_BVHCONSTRUCTOR_H

