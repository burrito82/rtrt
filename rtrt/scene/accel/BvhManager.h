#ifndef RTRT_SCENE_ACCEL_BVHMANAGER_H
#define RTRT_SCENE_ACCEL_BVHMANAGER_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "BvhBoundingBox.h"
#include "BvhNode.h"
#include "../TriangleGeometryDesc.h"
#include "../../cuda/VectorMemory.h"
/*============================================================================*/
/* DEFINES                                                                    */
/*============================================================================*/

/*============================================================================*/
/* FORWARD DECLARATIONS                                                       */
/*============================================================================*/
namespace rtrt
{
class Scene;
namespace bvh
{
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
class BvhManager
{
public:
    explicit BvhManager(Scene *pScene);

    std::vector<bvh::BvhBoundingBox> AddBvh(cuda::TriangleGeometryDesc &rTriangleGeometryDesc);
    void Synchronize();

    BvhNode *data();
    BvhNode *CudaPointer();

protected:
private:
    VectorMemory<BvhNode> m_vecBvh;
    VectorMemory<size_t> m_vecTriangleIndizes;

    Scene *m_pScene;
};

} // namespace bvh
} // namespace rtrt

#endif // ! RTRT_SCENE_ACCEL_BVHMANAGER_H

