#ifndef RTRT_SCENE_SCENE_H
#define RTRT_SCENE_SCENE_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "TriangleObject.h"
#include "../cuda/VectorMemory.h"
#include "../LibraryConfig.h"

#include <memory>
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

class SceneCuda;

class RTRTAPI Scene
{
public:
    Scene();

    void AddObject(TriangleObject const &rTriangleObject);
    void Synchronize();

    void Test();

private:
    std::shared_ptr<SceneCuda> m_pSceneCuda;

    VectorMemory<cuda::TriangleObjectDesc> m_vecTriangleObjects;
    VectorMemory<Point> m_vecPoints;
    VectorMemory<Normal> m_vecNormals;
};

} // namespace rtrt

#endif // ! RTRT_SCENE_SCENE_H

