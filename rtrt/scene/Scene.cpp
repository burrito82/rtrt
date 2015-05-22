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
#include <chrono>
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
    m_vecNormals{},
    m_vecBoundingBoxes{},
    m_oBvhManager{this}
{

}
/*============================================================================*/
/* IMPLEMENTATION                                                             */
/*============================================================================*/

void Scene::AddObject(TriangleObject const &rTriangleObject)
{
    auto iPointsNow = m_vecPoints.size();
    auto iNormalsNow = m_vecNormals.size();
    cuda::TriangleObjectDesc oTriangleObjDesc{};
    oTriangleObjDesc.m_iStartIndex = iPointsNow / 3u;
    oTriangleObjDesc.m_iNumberOfTriangles = rTriangleObject.m_vecPoints.size() / 3u;
    std::copy(std::begin(rTriangleObject.m_vecPoints), std::end(rTriangleObject.m_vecPoints), std::back_inserter(m_vecPoints));
    std::copy(std::begin(rTriangleObject.m_vecNormals), std::end(rTriangleObject.m_vecNormals), std::back_inserter(m_vecNormals));

    size_t const iBegin = oTriangleObjDesc.m_iStartIndex;
    size_t const iEnd = iBegin + oTriangleObjDesc.m_iNumberOfTriangles;
    for (size_t iTriangle = iBegin; iTriangle != iEnd; ++iTriangle)
    {
        cuda::TrianglePoints oTriangle = GetTrianglePoints(iTriangle);
        BoundingBox oBoundingBox{thrust::get<0>(oTriangle), thrust::get<0>(oTriangle)};
        oBoundingBox.Grow(BoundingBox{thrust::get<1>(oTriangle), thrust::get<1>(oTriangle)});
        oBoundingBox.Grow(BoundingBox{thrust::get<2>(oTriangle), thrust::get<2>(oTriangle)});
        m_vecBoundingBoxes.push_back(oBoundingBox);
    }

    m_oBvhManager.AddBvh(oTriangleObjDesc);
    m_vecTriangleObjects.push_back(oTriangleObjDesc);
}

void Scene::Synchronize()
{
    m_vecPoints.Synchronize();
    m_vecNormals.Synchronize();
    m_vecTriangleObjects.Synchronize();
    m_oBvhManager.Synchronize();
    m_pSceneCuda->Get().m_pPoints = m_vecPoints.CudaPointer();
    m_pSceneCuda->Get().m_pNormals = m_vecNormals.CudaPointer();
    m_pSceneCuda->Get().m_pTriangleObjects = m_vecTriangleObjects.CudaPointer();
    m_pSceneCuda->Get().m_iNumberOfTriangleObjects = m_vecTriangleObjects.size();
    m_pSceneCuda->Get().m_pBvhs = m_oBvhManager.CudaPointer();
    m_pSceneCuda->Synchronize();
}

void Scene::Test()
{
    cuda::KernelCheck();
    cuda::Scene oCpuScene{};
    oCpuScene.m_pPoints = m_vecPoints.data();
    oCpuScene.m_pNormals = m_vecNormals.data();
    oCpuScene.m_pTriangleObjects = m_vecTriangleObjects.data();
    oCpuScene.m_iNumberOfTriangleObjects = m_vecTriangleObjects.size();

    rtrt::VectorMemory<Ray> vecRays{};
    rtrt::VectorMemory<cuda::HitPoint, GPU_TO_CPU> vecHitPoints{};
    int yDim = 32;
    int xDim = 64;
    float z = 10;

    for (int y = -yDim; y <= yDim; ++y)
    {
        for (int x = -xDim; x <= xDim; ++x)
        {
            vecRays.push_back(Ray{Point{static_cast<float>(x) / (0.5f * xDim), -static_cast<float>(y) / (0.5f * yDim), z}, Normal{0.0f, 0.0f, -1.0f}});
        }
    }
    cuda::KernelCheck();
    vecRays.Synchronize();
    cuda::KernelCheck();

    // CPU
    auto startTime = std::chrono::system_clock::now();
    int iRay = 0;
    for (int y = -yDim; y <= yDim; ++y)
    {
        for (int x = -xDim; x <= xDim; ++x)
        {
            vecHitPoints.push_back(oCpuScene.Intersect(vecRays[iRay++]));
        }
    }
    std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count() << " ms" << std::endl;

    // DRAW
    iRay = 0;
    for (int y = -yDim; y <= yDim; ++y)
    {
        std::cout << '|';
        for (int x = -xDim; x <= xDim; ++x)
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

    std::cout << 'x';
    for (int x = -xDim; x <= xDim; ++x)
    {
        std::cout << '=';
    }
    std::cout << "x\n";

    // GPU
    startTime = std::chrono::system_clock::now();
    vecHitPoints.resize(vecRays.size());
    cuda::KernelCheck();
    unsigned int iNoRays = static_cast<unsigned int>(vecRays.size());
    unsigned int iThreadsPerBlock = 1024;
    dim3 blockDim{iThreadsPerBlock};
    dim3 gridDim{(iNoRays - 1) / iThreadsPerBlock + 1};
    using cuda::kernel::Raytrace;
    cuda::KernelCheck();
    Raytrace(blockDim, gridDim, m_pSceneCuda->CudaPointer(), vecRays.CudaPointer(), vecRays.size(), vecHitPoints.CudaPointer());
    cuda::KernelCheck();
    vecHitPoints.Synchronize();
    cuda::KernelCheck();
    std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count() << " ms" << std::endl;

    // DRAW
    iRay = 0;
    for (int y = -yDim; y <= yDim; ++y)
    {
        std::cout << '|';
        for (int x = -xDim; x <= xDim; ++x)
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

