#ifndef RTRT_SCENE_ACCEL_BVHMANAGER_H
#define RTRT_SCENE_ACCEL_BVHMANAGER_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "BvhNode.h"
#include "../Triangle.cuh"
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
    explicit BvhManager::BvhManager(Scene *pScene);

    void AddBvh(cuda::TriangleObjectDesc &oTriangleObjDesc);
    void Synchronize();

protected:
private:
    VectorMemory<BvhNode> m_vecBvh;

    Scene *m_pScene;
};

} // namespace bvh
} // namespace rtrt

#endif // ! RTRT_SCENE_ACCEL_BVHMANAGER_H

