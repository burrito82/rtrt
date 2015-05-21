#include "BvhManager.h"
/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "BvhBuilder.h"
#include "../Scene.h"

#include <algorithm>
#include <iterator>
/*============================================================================*/
/* MACROS AND DEFINES, CONSTANTS AND STATICS                                  */
/*============================================================================*/
namespace rtrt
{
namespace bvh
{
/*============================================================================*/
/* LOCAL VARS AND FUNCS                                                       */
/*============================================================================*/
namespace
{

class BvhTmpNode
{
public:
    BvhTmpNode(std::vector<size_t> &&vecTriangleIndizes, BoundingBox const &rBoundingBox):
        m_pLeft{nullptr},
        m_pRight{nullptr},
        m_oBoundingBox{rBoundingBox},
        m_vecTriangleIndizes(std::move(vecTriangleIndizes)),
        m_bIsLeaf{true}
    {

    }

    BvhTmpNode *Left()
    {
        return m_pLeft.get();
    }
    BvhTmpNode *Right()
    {
        return m_pRight.get();
    }

    void Construct()
    {
        size_t const iMaxBins = 64u;
        size_t iBins = std::min(iMaxBins, m_vecTriangleIndizes.size());
        if (m_vecTriangleIndizes.size() <= 2u)
        {
            return;
        }
        size_t iBiggestDim = m_oBoundingBox.MaximumExtent();
        std::vector<decltype(m_vecTriangleIndizes)::iterator> vecBinPartitions{iBins};
        std::vector<float> LeftArea{iBins};
        std::vector<float> RightArea{iBins};
        for (size_t iBin = 0u; iBin < iBins; ++iBin)
        {
            vecBinPartitions[iBin] = std::begin(m_vecTriangleIndizes) + iBin;
            std::sort(std::begin(m_vecTriangleIndizes), std::end(m_vecTriangleIndizes),
                      [&](auto i)
            {
                m_
            });
        }
    }

private:
    std::shared_ptr<BvhTmpNode> m_pLeft;
    std::shared_ptr<BvhTmpNode> m_pRight;
    BoundingBox m_oBoundingBox;
    std::vector<size_t> m_vecTriangleIndizes;
    bool m_bIsLeaf;
};

}
/*============================================================================*/
/* CONSTRUCTORS / DESTRUCTOR                                                  */
/*============================================================================*/
BvhManager::BvhManager(Scene *pScene):
                       m_vecBvh{},
                       m_pScene{pScene}
{

}
/*============================================================================*/
/* IMPLEMENTATION                                                             */
/*============================================================================*/
void BvhManager::AddBvh(cuda::TriangleObjectDesc &oTriangleObjDesc)
{
    BoundingBox oSceneBoundingBox{};
    for (auto const &p : m_pScene->GetPoints())
    {
        oSceneBoundingBox.Grow(BoundingBox{p, p});
    }
    std::vector<size_t> vecTriangleIndizes{};
    auto iBegin = oTriangleObjDesc.m_iStartIndex;
    auto iEnd = iBegin + oTriangleObjDesc.m_iNumberOfTriangles;
    for (auto i = iBegin; i != iEnd; ++i)
    {
        vecTriangleIndizes.push_back(i);
    }

    BvhTmpNode oBvhBuilding{std::move(vecTriangleIndizes), oSceneBoundingBox};
    oBvhBuilding.Construct();
}

void BvhManager::Synchronize()
{
    m_vecBvh.Synchronize();
}

} // namespace bvh
} // namespace rtrt

