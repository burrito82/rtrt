#ifndef RTRT_SCENE_SCENE_H
#define RTRT_SCENE_SCENE_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "BoundingBox.h"
#include "Material.h"
#include "PointLight.h"
#include "Triangle.h"
#include "TriangleGeometryDesc.h"
#include "TriangleGeometry.h"
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
namespace cuda
{
struct HitPoint;
}
/*============================================================================*/
/* STRUCT DEFINITIONS                                                         */
/*============================================================================*/

class SceneCuda;

class RTRTAPI Scene
{
    friend class bvh::BvhManager;
public:
    enum Hardware
    {
        CPU,
        GPU
    };

    Scene();

    std::size_t AddGeometry(TriangleGeometry const &rTriangleGeometry);
    std::size_t AddMaterial(Material const &rMaterial);
    std::size_t AddObject(TriangleObject oTriangleObject);
    std::size_t AddPointLight(PointLight const &rPointLight);

    void Synchronize();

    void Intersect(VectorMemory<Ray> const &rVecRays, 
                   VectorMemory<cuda::HitPoint, GPU_TO_CPU> &rVecHitPoints,
                   Hardware eHardware = GPU);

    std::vector<unsigned char> Test(int iWidth = 150, int iHeight = 80, Hardware eHardware = GPU, Matrix const &rMatTransformation = Matrix{});

    cuda::TrianglePoints GetTrianglePoints(std::size_t iTriangleIndex) const
    {
        return cuda::TrianglePoints
        {
            m_vecPoints[3 * iTriangleIndex],
            m_vecPoints[3 * iTriangleIndex + 1],
            m_vecPoints[3 * iTriangleIndex + 2]
        };
    }

    cuda::TriangleNormals GetTriangleNormals(std::size_t iTriangleIndex) const
    {
        return cuda::TriangleNormals
        {
            m_vecNormals[3 * iTriangleIndex],
            m_vecNormals[3 * iTriangleIndex + 1],
            m_vecNormals[3 * iTriangleIndex + 2]
        };
    }

    VectorMemory<cuda::TriangleGeometryDesc> const &GetTriangleGeometries() const
    {
        return m_vecTriangleGeometryDesc;
    }
    VectorMemory<TriangleObject> const &GetTriangleObjects() const
    {
        return m_vecTriangleObjects;
    }
    VectorMemory<TriangleObject> &GetTriangleObjects()
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

    VectorMemory<cuda::TriangleGeometryDesc> m_vecTriangleGeometryDesc;
    VectorMemory<Point> m_vecPoints;
    VectorMemory<Normal> m_vecNormals;
    bvh::BvhManager m_oBvhManager;

    VectorMemory<TriangleObject> m_vecTriangleObjects;
    VectorMemory<Material> m_vecMaterials;
    VectorMemory<PointLight> m_vecPointLights;
};

} // namespace rtrt

#endif // ! RTRT_SCENE_SCENE_H

