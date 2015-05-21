#ifndef RTRT_SCENE_SCENE_H
#define RTRT_SCENE_SCENE_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "TriangleObject.h"
#include "accel/BvhManager.h"
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
    friend class bvh::BvhManager;
public:
    Scene();

    void AddObject(TriangleObject const &rTriangleObject);
    void Synchronize();

    void Test();

    VectorMemory<cuda::TriangleObjectDesc> const &GetTriangleObjects() const
    {
        return m_vecTriangleObjects;
    }
    VectorMemory<Point> const &GetPoints() const
    {
        return m_vecPoints;
    }
    VectorMemory<Normal> const &GetNormals() const
    {
        return m_vecNormals;
    }

private:
    std::shared_ptr<SceneCuda> m_pSceneCuda;

    VectorMemory<cuda::TriangleObjectDesc> m_vecTriangleObjects;
    VectorMemory<Point> m_vecPoints;
    VectorMemory<Normal> m_vecNormals;
    bvh::BvhManager m_oBvhManager;
};

} // namespace rtrt

#endif // ! RTRT_SCENE_SCENE_H

