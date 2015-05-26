#include "Scene.h"
/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Scene.cuh"
#include "Triangle.h"
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
#include <tuple>
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
    m_vecTriangleGeometryDesc{},
    m_vecPoints{},
    m_vecNormals{},
    m_oBvhManager{this},
    m_vecTriangleObjects{},
    m_vecMaterials{},
    m_vecPointLights{}
{

}
/*============================================================================*/
/* IMPLEMENTATION                                                             */
/*============================================================================*/

size_t Scene::AddGeometry(TriangleGeometry const &rTriangleGeometry)
{
    Assert(rTriangleGeometry.m_vecPoints.size() == rTriangleGeometry.m_vecNormals.size(), "Every vertex must have a normal!");
    auto iPointsNow = m_vecPoints.size();
    cuda::TriangleGeometryDesc oTriangleObjDesc{};
    oTriangleObjDesc.m_iStartIndex = iPointsNow / 3u;
    oTriangleObjDesc.m_iNumberOfTriangles = rTriangleGeometry.m_vecPoints.size() / 3u;
    std::copy(std::begin(rTriangleGeometry.m_vecPoints), std::end(rTriangleGeometry.m_vecPoints), std::back_inserter(m_vecPoints));
    std::copy(std::begin(rTriangleGeometry.m_vecNormals), std::end(rTriangleGeometry.m_vecNormals), std::back_inserter(m_vecNormals));

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
    m_vecTriangleGeometryDesc.push_back(oTriangleObjDesc);
    return m_vecTriangleGeometryDesc.size() - 1u;
}

size_t Scene::AddMaterial(Material const &rMaterial)
{
    m_vecMaterials.push_back(rMaterial);
    return m_vecMaterials.size() - 1u;
}

size_t Scene::AddObject(TriangleObject oTriangleObject)
{
    oTriangleObject.m_matInvTransformation = oTriangleObject.m_matTransformation.Inverted();
    m_vecTriangleObjects.push_back(oTriangleObject);
    return m_vecTriangleObjects.size() - 1u;
}

size_t Scene::AddPointLight(PointLight const &rPointLight)
{
    m_vecPointLights.push_back(rPointLight);
    return m_vecPointLights.size() - 1u;
}

void Scene::Synchronize()
{
    m_vecPoints.Synchronize();
    m_vecNormals.Synchronize();
    m_vecTriangleGeometryDesc.Synchronize();
    m_oBvhManager.Synchronize();
    m_vecTriangleObjects.Synchronize();
    m_vecMaterials.Synchronize();
    m_vecPointLights.Synchronize();
    m_pSceneCuda->Get().m_pPoints = m_vecPoints.CudaPointer();
    m_pSceneCuda->Get().m_pNormals = m_vecNormals.CudaPointer();
    m_pSceneCuda->Get().m_pTriangleGeometryDesc = m_vecTriangleGeometryDesc.CudaPointer();
    m_pSceneCuda->Get().m_pBvhs = m_oBvhManager.CudaPointer();
    m_pSceneCuda->Get().m_iNumberOfTriangleObjects = m_vecTriangleObjects.size();
    m_pSceneCuda->Get().m_pTriangleObjects = m_vecTriangleObjects.CudaPointer();
    m_pSceneCuda->Get().m_pMaterials = m_vecMaterials.CudaPointer();
    m_pSceneCuda->Get().m_pPointLights = m_vecPointLights.CudaPointer();
    m_pSceneCuda->Synchronize();
}

std::vector<unsigned char> Scene::Test(int iWidth, int iHeight, Hardware eHardware, Matrix const &rMatTransformation)
{
    cuda::TypeVerifier::VerifySize();
    cuda::KernelCheck();

    static rtrt::VectorMemory<Ray> vecRays{};
    static rtrt::VectorMemory<cuda::HitPoint, GPU_TO_CPU> vecHitPoints{};
    float z = 0.0f;
    bool bDraw = (iWidth < 160);

    //if (vecRays.size() != iWidth * iHeight)
    {
        vecRays.resize(iWidth * iHeight);
        float fCameraExtentX = 4.0f;
        float fCameraExtentY = fCameraExtentX / iWidth * iHeight;
        for (int yStep = 0; yStep < iHeight; ++yStep)
        {
#pragma omp parallel for
            for (int xStep = 0; xStep < iWidth; ++xStep)
            {
                /*vecRays[xStep + yStep * iWidth] = Ray{Point{rMatTransformation *
                    Point
                    {
                        fCameraExtentX * (static_cast<float>(xStep) / static_cast<float>(iWidth) - 0.5f),
                        -fCameraExtentY * (static_cast<float>(yStep) / static_cast<float>(iHeight) - 0.5f),
                        z
                    }},
                    Normal{rMatTransformation * Normal{
                    {
                        0.0f,
                        0.0f,
                        -1.0f
                    }}}
                };*/
                vecRays[xStep + yStep * iWidth] = Ray{Point{rMatTransformation *
                    Point
                    {
                        0.0f,
                        0.0f,
                        z
                    }},
                    Normal{rMatTransformation * Normal{
                    {
                        fCameraExtentX * (static_cast<float>(xStep) / static_cast<float>(iWidth)-0.5f),
                        -fCameraExtentY * (static_cast<float>(yStep) / static_cast<float>(iHeight)-0.5f),
                        -2.0f
                    }}}
                };
            }
        }
    }

    auto startTime = std::chrono::system_clock::time_point{};
    auto duration = std::chrono::system_clock::duration{};
    if (eHardware == CPU)
    {
        // CPU
        startTime = std::chrono::system_clock::now();
        Intersect(vecRays, vecHitPoints, CPU);
        duration = std::chrono::system_clock::now() - startTime;
    }

    if (eHardware == GPU)
    {
        // GPU
        cuda::KernelCheck();
        vecRays.Synchronize();
        cuda::KernelCheck();
        startTime = std::chrono::system_clock::now();
        Intersect(vecRays, vecHitPoints, GPU);
        duration = std::chrono::system_clock::now() - startTime;
    }
    auto iIntersectionTime = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    auto iNumberOfAllRays = vecRays.size();

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
        vecResult[4 * i + 3] = 0xff;
    }
    using SecondaryRay = std::tuple<Ray, int, int, float>;
    std::vector<std::vector<SecondaryRay>> vecVecSecondaryRays(iHeight);
#pragma omp parallel for
    for (auto i = 0; i < vecHitPoints.size(); ++i)
    {
        auto const &eye = vecRays[i];
        auto const &hit = vecHitPoints[i];
        for (auto iLight = 0; iLight < m_vecPointLights.size(); ++iLight)
        {
            auto const &rLightSource = m_vecPointLights[iLight];
            if (hit)
            {
                float afLightColor[] = {
                    rLightSource.color.r,
                    rLightSource.color.g,
                    rLightSource.color.b,
                    rLightSource.color.a,
                };
                float afMatColor[] = {
                    m_vecMaterials[m_vecTriangleObjects[hit.m_iObjectIndex].m_iMaterial].diffuse.r,
                    m_vecMaterials[m_vecTriangleObjects[hit.m_iObjectIndex].m_iMaterial].diffuse.g,
                    m_vecMaterials[m_vecTriangleObjects[hit.m_iObjectIndex].m_iMaterial].diffuse.b,
                    m_vecMaterials[m_vecTriangleObjects[hit.m_iObjectIndex].m_iMaterial].diffuse.a,
                };
                for (auto iPixel = 4 * i; iPixel < 4 * i + 3; ++iPixel)
                {
                    vecResult[iPixel] = static_cast<unsigned char>(0xff & std::min(0xff, std::max<int>(0, static_cast<int>(static_cast<float>(
                        std::max(0.0f, static_cast<float>(vecResult[iPixel])
                        + afLightColor[iPixel % 4] * afMatColor[iPixel % 4] * Dot(hit.n, Normalized(rLightSource.p - hit.p) / Length(rLightSource.p - hit.p))))))));
                }
            }
        }
        float fReflectivity = m_vecMaterials[m_vecTriangleObjects[hit.m_iObjectIndex].m_iMaterial].m_fReflectivity;
        if (hit && fReflectivity > 0.0f)
        {
            auto x = i % iWidth;
            auto y = i / iWidth;
            vecVecSecondaryRays[y].emplace_back(Ray{hit.p + 0.001f * hit.n, Reflect(vecRays[i].direction, hit.n)}, x, y, fReflectivity);
        }
    }
    int const iMaxRecursion = 3;
    for (int iCurrentRecursionStep = 0; iCurrentRecursionStep < iMaxRecursion; ++iCurrentRecursionStep)
    {
        std::vector<SecondaryRay> vecSecondaryRays{};
        for (auto const &a : vecVecSecondaryRays)
        {
            std::move(std::begin(a), std::end(a), std::back_inserter(vecSecondaryRays));
        }
        if (!vecSecondaryRays.empty())
        {
            VectorMemory<Ray> vecAllSecondaryRays(vecSecondaryRays.size());
            std::transform(std::begin(vecSecondaryRays), std::end(vecSecondaryRays), std::begin(vecAllSecondaryRays),
                           [](SecondaryRay const &t)
            {
                return std::get<0>(t);
            });
            if (eHardware == GPU)
            {
                vecAllSecondaryRays.Synchronize();
            }
            VectorMemory<cuda::HitPoint, GPU_TO_CPU> vecSecondaryHitPoints(vecAllSecondaryRays.size());

            auto startTimeSR = std::chrono::system_clock::now();
            Intersect(vecAllSecondaryRays, vecSecondaryHitPoints, eHardware);
            iIntersectionTime += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTimeSR).count();
            iNumberOfAllRays += vecRays.size();
#pragma omp parallel for
            for (int i = 0; i < vecSecondaryRays.size(); ++i)
            {
                auto const &eye = vecAllSecondaryRays[i];
                auto const &hit = vecSecondaryHitPoints[i];
                if (hit)
                {
                    int x = std::get<1>(vecSecondaryRays[i]);
                    int y = std::get<2>(vecSecondaryRays[i]);
                    float fOldReflectivity = std::get<3>(vecSecondaryRays[i]);
                    for (auto const &rLightSource : m_vecPointLights)
                    {
                        float afLightColor[] = {
                            rLightSource.color.r,
                            rLightSource.color.g,
                            rLightSource.color.b,
                            rLightSource.color.a,
                        };
                        float afMatColor[] = {
                            m_vecMaterials[m_vecTriangleObjects[hit.m_iObjectIndex].m_iMaterial].diffuse.r,
                            m_vecMaterials[m_vecTriangleObjects[hit.m_iObjectIndex].m_iMaterial].diffuse.g,
                            m_vecMaterials[m_vecTriangleObjects[hit.m_iObjectIndex].m_iMaterial].diffuse.b,
                            m_vecMaterials[m_vecTriangleObjects[hit.m_iObjectIndex].m_iMaterial].diffuse.a,
                        };
                        for (auto iPixel = 4 * (x + y * iWidth); iPixel < 4 * (x + y * iWidth) + 3; ++iPixel)
                        {
                            float fOldColor = static_cast<float>(vecResult[iPixel]);
                            float fNewColor = afLightColor[iPixel % 4] * afMatColor[iPixel % 4] * std::max(0.0f, Dot(hit.n, Normalized(rLightSource.p - hit.p)) / Length(rLightSource.p - hit.p));
                            vecResult[iPixel] = static_cast<unsigned char>(0xff & std::min(0xff, std::max<int>(0, static_cast<int>(static_cast<float>(
                                std::max(0.0f,
                                (1.0f - fOldReflectivity) * fOldColor
                                + fOldReflectivity * fNewColor))))));
                        }
                    }
                    float fReflectivity = m_vecMaterials[m_vecTriangleObjects[hit.m_iObjectIndex].m_iMaterial].m_fReflectivity;
                    if (hit && fReflectivity > 0.0f)
                    {
                        vecVecSecondaryRays[y].emplace_back(Ray{hit.p + 0.001f * hit.n, Reflect(vecAllSecondaryRays[i].direction, hit.n)}, x, y, fOldReflectivity * fReflectivity);
                    }
                }
            }
        }
    }

    std::cout << "Time elapsed: " << iIntersectionTime << " ms\t("
        << (static_cast<double>(iNumberOfAllRays / 1000u) / iIntersectionTime) << "M rays/s,\t"
        << iNumberOfAllRays << " rays,\t"
        << (m_vecPoints.size() / 3u) << " triangles)" << std::endl;

    return vecResult;
}

void Scene::Intersect(VectorMemory<Ray> const &rVecRays,
                      VectorMemory<cuda::HitPoint, GPU_TO_CPU> &rVecHitPoints,
                      Hardware eHardware)
{
    if (rVecRays.empty())
        return;
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
        oCpuScene.m_pTriangleGeometryDesc = m_vecTriangleGeometryDesc.data();
        oCpuScene.m_pPoints = m_vecPoints.data();
        oCpuScene.m_pNormals = m_vecNormals.data();
        oCpuScene.m_pBvhs = m_oBvhManager.data();
        oCpuScene.m_iNumberOfTriangleObjects = m_vecTriangleObjects.size();
        oCpuScene.m_pTriangleObjects = m_vecTriangleObjects.data();
        oCpuScene.m_pMaterials = m_vecMaterials.data();
        oCpuScene.m_pPointLights = m_vecPointLights.data();

        int iRaysEnd = static_cast<int>(rVecRays.size());
#pragma omp parallel for
        for (int i = 0; i < iRaysEnd; ++i)
        {
            rVecHitPoints[i] = oCpuScene.Intersect(rVecRays[i]);
        }
    }
}

} // namespace rtrt

