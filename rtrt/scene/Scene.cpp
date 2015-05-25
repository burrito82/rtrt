#include "Scene.h"
/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Scene.cuh"
#include "Triangle.cuh"
#include "../Assert.h"
#include "../cuda/Device.h"
#include "../cuda/TypeVerifier.h"
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
    m_oBvhManager{this},
    m_vecMaterials{},
    m_vecPointLights{}
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

void Scene::AddPointLight(PointLight const &rPointLight)
{
    m_vecPointLights.push_back(rPointLight);
}

void Scene::Synchronize()
{
    m_vecPoints.Synchronize();
    m_vecNormals.Synchronize();
    m_vecTriangleObjects.Synchronize();
    m_oBvhManager.Synchronize();
    m_vecMaterials.Synchronize();
    m_vecPointLights.Synchronize();
    m_pSceneCuda->Get().m_pPoints = m_vecPoints.CudaPointer();
    m_pSceneCuda->Get().m_pNormals = m_vecNormals.CudaPointer();
    m_pSceneCuda->Get().m_pTriangleObjects = m_vecTriangleObjects.CudaPointer();
    m_pSceneCuda->Get().m_iNumberOfTriangleObjects = m_vecTriangleObjects.size();
    m_pSceneCuda->Get().m_pBvhs = m_oBvhManager.CudaPointer();
    m_pSceneCuda->Get().m_pMaterials = m_vecMaterials.CudaPointer();
    m_pSceneCuda->Get().m_pPointLights = m_vecPointLights.CudaPointer();
    m_pSceneCuda->Synchronize();
}

std::vector<unsigned char> Scene::Test(int iWidth, int iHeight, Hardware eHardware)
{
    cuda::TypeVerifier::VerifySize();
    cuda::KernelCheck();
    cuda::Scene oCpuScene{};
    oCpuScene.m_iNumberOfTriangleObjects = m_vecTriangleObjects.size();
    oCpuScene.m_pTriangleObjects = m_vecTriangleObjects.data();
    oCpuScene.m_pPoints = m_vecPoints.data();
    oCpuScene.m_pNormals = m_vecNormals.data();
    oCpuScene.m_pBvhs = m_oBvhManager.data();

    static rtrt::VectorMemory<Ray> vecRays{};
    static rtrt::VectorMemory<cuda::HitPoint, GPU_TO_CPU> vecHitPoints{};
    float z = 10;
    bool bDraw = (iWidth < 160);

    if (vecRays.size() != iWidth * iHeight)
    {
        vecRays.resize(iWidth * iHeight);
        float fCameraExtentX = 4.0f;
        float fCameraExtentY = fCameraExtentX / iWidth * iHeight;
        for (int yStep = 0; yStep < iHeight; ++yStep)
        {
#pragma omp parallel for
            for (int xStep = 0; xStep < iWidth; ++xStep)
            {
                vecRays[xStep + yStep * iWidth] = Ray{Point{
                    0.0f,
                    0.0f,
                    z},
                    Normal{
                        fCameraExtentX * (static_cast<float>(xStep) / static_cast<float>(iWidth)-0.5f),
                        -fCameraExtentY * (static_cast<float>(yStep) / static_cast<float>(iHeight)-0.5f),
                        -2.0f}};
            }
        }
    }

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
    }

    if (eHardware == GPU)
    {
        // GPU
        cuda::KernelCheck();
        vecRays.Synchronize();
        cuda::KernelCheck();
        auto startTime = std::chrono::system_clock::now();
        Intersect(vecRays, vecHitPoints, GPU);
        auto duration = std::chrono::system_clock::now() - startTime;
        std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << " ms\t("
            << (static_cast<double>(vecRays.size() / 1000u) / static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count())) << "M rays/s,\t"
            << vecRays.size() << " rays,\t"
            << (m_vecPoints.size() / 3u) << " triangles)" << std::endl;
    }

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

    static std::vector<unsigned char> vecResult{};
    vecResult.resize(vecHitPoints.size() * 4);
    std::fill(std::begin(vecResult), std::end(vecResult), 0);
#pragma omp parallel for
    for (auto i = 0; i < vecHitPoints.size(); ++i)
    {
        auto const &hit = vecHitPoints[i];
        //vecResult[4 * i] = static_cast<unsigned char>(std::min(0xff, std::max<int>(0, static_cast<int>((255.0f - hit.m_fDistance) * hit.n[0]))));
        //vecResult[4 * i + 1] = static_cast<unsigned char>(std::min(0xff, std::max<int>(0, static_cast<int>((255.0f - hit.m_fDistance) * hit.n[1]))));
        //vecResult[4 * i + 2] = static_cast<unsigned char>(std::min(0xff, std::max<int>(0, static_cast<int>((255.0f - hit.m_fDistance) * hit.n[2]))));
        vecResult[4 * i + 3] = 0xff;
    }
    for (auto const &rLightSource : m_vecPointLights)
    {
#pragma omp parallel for
        for (auto i = 0; i < vecHitPoints.size(); ++i)
        {
            auto const &eye = vecRays[i];
            auto const &hit = vecHitPoints[i];
            if (hit)
            {
                float const *color = &rLightSource.color.r;
                for (auto x = 4 * i; x < 4 * i + 2; ++x)
                {
                    vecResult[x] = static_cast<unsigned char>(std::min(0xff, std::max<int>(0, static_cast<int>(static_cast<float>(
                        std::max(0.0f, static_cast<float>(vecResult[x])
                        + color[x % 4] * Dot(hit.n, Normalized(rLightSource.p - hit.p) / Length(rLightSource.p - hit.p))))))));
                }
            }
        }
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
        cuda::kernel::Raytrace(m_pSceneCuda->CudaPointer(), rVecRays.CudaPointer(), rVecRays.size(), rVecHitPoints.CudaPointer());
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
            rVecHitPoints[i] = oCpuScene.Intersect(rVecRays[i]);
        }
    }
}

} // namespace rtrt

