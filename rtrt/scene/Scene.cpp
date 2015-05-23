#include "Scene.h"
/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Scene.cuh"
#include "Triangle.cuh"
#include "../Assert.h"
#include "../cuda/Device.h"
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
    m_oBvhManager{this}
{

}
/*============================================================================*/
/* IMPLEMENTATION                                                             */
/*============================================================================*/

void Scene::AddObject(TriangleObject const &rTriangleObject)
{
    Assert(rTriangleObject.m_vecPoints.size() == rTriangleObject.m_vecNormals.size(), "Every vertex must have a normal!");
    auto iPointsNow = m_vecPoints.size();
    cuda::TriangleObjectDesc oTriangleObjDesc{};
    oTriangleObjDesc.m_iStartIndex = iPointsNow / 3u;
    oTriangleObjDesc.m_iNumberOfTriangles = rTriangleObject.m_vecPoints.size() / 3u;
    std::copy(std::begin(rTriangleObject.m_vecPoints), std::end(rTriangleObject.m_vecPoints), std::back_inserter(m_vecPoints));
    std::copy(std::begin(rTriangleObject.m_vecNormals), std::end(rTriangleObject.m_vecNormals), std::back_inserter(m_vecNormals));

    std::vector<bvh::BvhBoundingBox> vecBoundingBoxes = m_oBvhManager.AddBvh(oTriangleObjDesc);
    std::vector<Point> vecPoints(m_vecPoints.size() - iPointsNow);
    std::vector<Normal> vecNormals(m_vecNormals.size() - iPointsNow);
    auto it = std::begin(vecBoundingBoxes);
    for (size_t i = oTriangleObjDesc.m_iStartIndex; it != std::end(vecBoundingBoxes); ++i, ++it)
    {
        vecPoints[3u * i - iPointsNow] = m_vecPoints[3u * it->m_iTriangleIndex + iPointsNow];
        vecPoints[3u * i - iPointsNow + 1] = m_vecPoints[3u * it->m_iTriangleIndex + iPointsNow + 1u];
        vecPoints[3u * i - iPointsNow + 2] = m_vecPoints[3u * it->m_iTriangleIndex + iPointsNow + 2u];
        vecNormals[3u * i - iPointsNow] = m_vecNormals[3u * it->m_iTriangleIndex + iPointsNow];
        vecNormals[3u * i - iPointsNow + 1] = m_vecNormals[3u * it->m_iTriangleIndex + iPointsNow + 1];
        vecNormals[3u * i - iPointsNow + 2] = m_vecNormals[3u * it->m_iTriangleIndex + iPointsNow + 2];
    }
    std::copy(std::begin(vecPoints), std::end(vecPoints), std::begin(m_vecPoints) + iPointsNow);
    std::copy(std::begin(vecNormals), std::end(vecNormals), std::begin(m_vecNormals) + iPointsNow);
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

std::vector<unsigned char> Scene::Test(int iWidth, int iHeight, Hardware eHardware)
{
    cuda::KernelCheck();
    cuda::Scene oCpuScene{};
    oCpuScene.m_iNumberOfTriangleObjects = m_vecTriangleObjects.size();
    oCpuScene.m_pTriangleObjects = m_vecTriangleObjects.data();
    oCpuScene.m_pPoints = m_vecPoints.data();
    oCpuScene.m_pNormals = m_vecNormals.data();
    oCpuScene.m_pBvhs = m_oBvhManager.data();

    rtrt::VectorMemory<Ray> vecRays{};
    rtrt::VectorMemory<cuda::HitPoint, GPU_TO_CPU> vecHitPoints{};
    float z = 10;
    bool bDraw = (iWidth < 160);

    float fCameraExtentX = 4.0f;
    float fCameraExtentY = 4.0f;
    for (int yStep = 0; yStep < iHeight; ++yStep)
    {
        for (int xStep = 0; xStep < iWidth; ++xStep)
        {
            vecRays.push_back(Ray{Point{
                fCameraExtentX * (static_cast<float>(xStep) / static_cast<float>(iWidth) - 0.5f),
                -fCameraExtentY * (static_cast<float>(yStep) / static_cast<float>(iHeight) - 0.5f),
                z}, 
                Normal{0.0f, 0.0f, -1.0f}});
        }
    }
    cuda::KernelCheck();
    vecRays.Synchronize();
    cuda::KernelCheck();

    if (eHardware == CPU)
    {
        // CPU
        auto startTime = std::chrono::system_clock::now();
        Intersect(vecRays, vecHitPoints, CPU);
        auto duration = std::chrono::system_clock::now() - startTime;
        std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << " ms\t("
            << (static_cast<double>(vecRays.size() / 1000u) / static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count())) << "M rays/s,\t"
            << vecRays.size() << " rays,\t"
            << (m_vecPoints.size() / 3u) << " triangles)" << std::endl;

        // DRAW
        if (bDraw)
        {
            int iRay = 0;
            for (int y = 0; y < iHeight; ++y)
            {
                std::cout << '|';
                for (int x = 0; x < iWidth; ++x)
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
            for (int x = 0; x < iWidth; ++x)
            {
                std::cout << '=';
            }
            std::cout << "x\n";
        }
    }

    if (eHardware == GPU)
    {
        // GPU
        auto startTime = std::chrono::system_clock::now();
        Intersect(vecRays, vecHitPoints, GPU);
        auto duration = std::chrono::system_clock::now() - startTime;
        std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << " ms\t("
            << (static_cast<double>(vecRays.size() / 1000u) / static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count())) << "M rays/s,\t"
            << vecRays.size() << " rays,\t"
            << (m_vecPoints.size() / 3u) << " triangles)" << std::endl;

        // DRAW
        if (bDraw)
        {
            int iRay = 0;
            for (int y = 0; y < iHeight; ++y)
            {
                std::cout << '|';
                for (int x = 0; x < iWidth; ++x)
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
    }

    std::vector<unsigned char> vecResult{};
    for (auto const &hit : vecHitPoints)
    {
        unsigned char c = 0;
        if (!std::isinf(hit.m_fDistance))
        {
            c = static_cast<unsigned char>(std::min(0xff, std::max<int>(0, 255 - static_cast<int>(hit.m_fDistance))));
        }
        vecResult.insert(std::end(vecResult), {c, c, c, 0xff});
    }
    return vecResult;
}

void Scene::Intersect(VectorMemory<Ray> const &rVecRays,
                      VectorMemory<cuda::HitPoint, GPU_TO_CPU> &rVecHitPoints,
                      Hardware eHardware)
{
    rVecHitPoints.resize(rVecRays.size());
    if (eHardware == GPU)
    {
        cuda::KernelCheck();
        unsigned int iNoRays = static_cast<unsigned int>(rVecRays.size());
        unsigned int iThreadsPerBlock = cuda::Devices::GetInstance().Current().maxThreadsDim[0];
        unsigned int iGridSize = std::min<unsigned int>((iNoRays - 1u) / iThreadsPerBlock + 1u, cuda::Devices::GetInstance().Current().maxGridSize[0]);
        dim3 i3BlockDim{iThreadsPerBlock};
        dim3 i3GridDim{iGridSize};
        cuda::KernelCheck();
        cuda::kernel::Raytrace(i3BlockDim, i3GridDim, m_pSceneCuda->CudaPointer(), rVecRays.CudaPointer(), rVecRays.size(), rVecHitPoints.CudaPointer());
        cuda::KernelCheck();
        rVecHitPoints.Synchronize();
        cuda::KernelCheck();
    }
    else
    {
        cuda::Scene oCpuScene{};
        oCpuScene.m_iNumberOfTriangleObjects = m_vecTriangleObjects.size();
        oCpuScene.m_pTriangleObjects = m_vecTriangleObjects.data();
        oCpuScene.m_pPoints = m_vecPoints.data();
        oCpuScene.m_pNormals = m_vecNormals.data();
        oCpuScene.m_pBvhs = m_oBvhManager.data();

        int iRaysEnd = static_cast<int>(rVecRays.size());
#pragma omp parallel for
        for (int i = 0; i < iRaysEnd; ++i)
        {
            rVecHitPoints[i] = oCpuScene.IntersectBvh(rVecRays[i]);
        }
    }
}

void Scene::Raytrace()
{

}

} // namespace rtrt

