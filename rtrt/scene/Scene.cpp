#include "Scene.h"
/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Scene.cuh"
#include "Triangle.cuh"
#include "../cuda/VectorMemory.h"

#include <algorithm>
#include <iterator>

// REMOVE
#include <cmath>
#include <iostream>
#include <vector>
/*============================================================================*/
/* MACROS AND DEFINES, CONSTANTS AND STATICS                                  */
/*============================================================================*/
namespace rtrt
{
/*============================================================================*/
/* LOCAL VARS AND FUNCS                                                       */
/*============================================================================*/

class SceneCuda
{
public:
    SceneCuda():
        m_cuSceneCuda{1u},
        m_rSceneCuda{m_cuSceneCuda[0]}
    {

    }

    cuda::Scene &operator*()
    {
        return m_rSceneCuda;
    }

    cuda::Scene *operator->()
    {
        return &**this;
    }

    cuda::Scene &Get()
    {
        return m_rSceneCuda;
    }

    cuda::Scene *CudaPointer()
    {
        return m_cuSceneCuda.CudaPointer();
    }

    void Synchronize()
    {
        m_cuSceneCuda.Synchronize();
    }

private:
    VectorMemory<cuda::Scene> m_cuSceneCuda;
    cuda::Scene &m_rSceneCuda;
};

/*============================================================================*/
/* CONSTRUCTORS / DESTRUCTOR                                                  */
/*============================================================================*/
Scene::Scene():
    m_pSceneCuda{std::make_shared<SceneCuda>()},
    m_vecTriangleObjects{},
    m_vecPoints{},
    m_vecNormals{}
{

}
/*============================================================================*/
/* IMPLEMENTATION                                                             */
/*============================================================================*/

void Scene::AddObject(TriangleObject const &rTriangleObject)
{
    auto iPointsNow = m_vecPoints.size();
    auto iNormalsNow = m_vecNormals.size();
    cuda::TriangleObjectDesc oTriangleDesc{};
    oTriangleDesc.m_iStartIndex = iPointsNow / 3u;
    oTriangleDesc.m_iNumberOfTriangles = rTriangleObject.m_vecPoints.size() / 3u;
    std::copy(std::begin(rTriangleObject.m_vecPoints), std::end(rTriangleObject.m_vecPoints), std::back_inserter(m_vecPoints));
    std::copy(std::begin(rTriangleObject.m_vecNormals), std::end(rTriangleObject.m_vecNormals), std::back_inserter(m_vecNormals));
    m_vecTriangleObjects.push_back(oTriangleDesc);
}

void Scene::Synchronize()
{
    m_vecPoints.Synchronize();
    m_vecNormals.Synchronize();
    m_vecTriangleObjects.Synchronize();
    (*m_pSceneCuda)->m_pPoints = m_vecPoints.CudaPointer();
    (*m_pSceneCuda)->m_pNormals = m_vecNormals.CudaPointer();
    (*m_pSceneCuda)->m_pTriangleObjects = m_vecTriangleObjects.CudaPointer();
    (*m_pSceneCuda)->m_iNumberOfTriangleObjects = m_vecTriangleObjects.size();
    m_pSceneCuda->Synchronize();
}

void Scene::Test()
{
    cuda::Scene oCpuScene{};
    oCpuScene.m_pPoints = m_vecPoints.data();
    oCpuScene.m_pNormals = m_vecNormals.data();
    oCpuScene.m_pTriangleObjects = m_vecTriangleObjects.data();
    oCpuScene.m_iNumberOfTriangleObjects = m_vecTriangleObjects.size();

    rtrt::VectorMemory<Ray> vecRays{};
    rtrt::VectorMemory<cuda::HitPoint, GPU_TO_CPU> vecHitPoints{};

    for (int y = -25; y <= 25; ++y)
    {
        for (int x = -50; x <= 50; ++x)
        {
            vecRays.push_back(Ray{Point{static_cast<float>(x) / 25.0f, -static_cast<float>(y) / 12.5f, 15.0f}, Normal{0.0f, 0.0f, -1.0f}});
        }
    }
    vecRays.Synchronize();

    // CPU
    int iRay = 0;
    for (int y = -25; y <= 25; ++y)
    {
        for (int x = -50; x <= 50; ++x)
        {
            vecHitPoints.push_back(oCpuScene.Intersect(vecRays[iRay++]));
        }
    }

    // DRAW
    iRay = 0;
    for (int y = -25; y <= 25; ++y)
    {
        std::cout << '|';
        for (int x = -50; x <= 50; ++x)
        {
            cuda::HitPoint const &hitpoint = vecHitPoints[iRay++];
            if (hitpoint)
            {
                std::cout << "#";
            }
            else
            {
                std::cout << " ";
            }
        }
        std::cout << "|\n";
    }

    for (int x = -50; x <= 50; ++x)
    {
        std::cout << '=';
    }
    std::cout << '\n';

    // GPU
    vecHitPoints.resize(vecRays.size());
    dim3 blockDim{static_cast<unsigned int>(vecRays.size())};
    dim3 gridDim{};
    using cuda::kernel::Raytrace;
    Raytrace(blockDim, gridDim, m_pSceneCuda->CudaPointer(), vecRays.CudaPointer(), vecHitPoints.CudaPointer());
    cuda::Checked(cudaDeviceSynchronize());
    vecHitPoints.Synchronize();

    // DRAW
    iRay = 0;
    for (int y = -25; y <= 25; ++y)
    {
        std::cout << '|';
        for (int x = -50; x <= 50; ++x)
        {
            cuda::HitPoint const &hitpoint = vecHitPoints[iRay++];
            if (hitpoint)
            {
                std::cout << "#";
            }
            else
            {
                std::cout << " ";
            }
        }
        std::cout << "|\n";
    }
}

} // namespace rtrt

