#include "BvhManager.h"
/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "BvhBuilder.h"
#include "../Scene.h"
#include "../../Assert.h"

#include <algorithm>
#include <iterator>
#include <queue>
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
    BvhTmpNode(VectorMemory<BoundingBox>::iterator const itBegin,
               VectorMemory<BoundingBox>::iterator const itEnd,
               BoundingBox const &rBoundingBox):
        m_itBegin{itBegin},
        m_itEnd{itEnd},
        m_pLeft{nullptr},
        m_pRight{nullptr},
        m_oBoundingBox{rBoundingBox},
        m_bIsLeaf{true}
    {

    }

    BvhTmpNode const *Left() const
    {
        return m_pLeft.get();
    }
    BvhTmpNode const *Right() const
    {
        return m_pRight.get();
    }

    void Construct()
    {
        size_t const iNumberOfTriangles = std::distance(m_itBegin, m_itEnd);
        if (iNumberOfTriangles <= 2u)
        {
            return;
        }
        size_t const iMaxBins = 4u;
        size_t iBins = std::min(iMaxBins, iNumberOfTriangles);
        size_t iBiggestDim = m_oBoundingBox.MaximumExtent();

        // sort along longest axis
        std::sort(m_itBegin, m_itEnd,
                  [iBiggestDim](BoundingBox const &lhs, BoundingBox const &rhs)
        {
            return lhs.Center()[iBiggestDim] < rhs.Center()[iBiggestDim];
        });

        std::vector<decltype(m_itEnd)> vecBinIterators(iBins + 1);
        std::vector<BoundingBox> vecLeftBoundingBoxes(iBins);
        std::vector<BoundingBox> vecRightBoundingBoxes(iBins);
        std::vector<float> vecSAH(iBins);

        for (size_t iBin = 0u; iBin <= iBins; ++iBin)
        {
            size_t iBinEnd = (iBin * iNumberOfTriangles) / iBins;
            vecBinIterators[iBin] = m_itBegin + iBinEnd;
        }

        // compute left area
        BoundingBox oBinBoundingBox{};
        for (size_t iBin = 0u; iBin < iBins; ++iBin)
        {
            for (auto itTriangleBox = vecBinIterators[iBin]; itTriangleBox != vecBinIterators[iBin + 1]; ++itTriangleBox)
            {
                oBinBoundingBox.Grow(*itTriangleBox);
            }
            vecLeftBoundingBoxes[iBin] = oBinBoundingBox;
        }

        oBinBoundingBox = BoundingBox{};
        for (int iBin = static_cast<int>(iBins) - 1; iBin >= 0; --iBin)
        {
            for (auto itTriangleBox = vecBinIterators[iBin]; itTriangleBox != vecBinIterators[iBin + 1]; ++itTriangleBox)
            {
                oBinBoundingBox.Grow(*itTriangleBox);
            }
            vecRightBoundingBoxes[iBin] = oBinBoundingBox;
            vecSAH[iBin] = std::distance(m_itBegin, vecBinIterators[iBin + 1]) * vecLeftBoundingBoxes[iBin].HalfSurfaceArea()
                + std::distance(vecBinIterators[iBin + 1], m_itEnd) * vecRightBoundingBoxes[iBin].HalfSurfaceArea();
        }

        size_t iBestBin = 0u;
        float fBestSah = std::numeric_limits<float>::infinity();
        for (size_t iBin = 0u; iBin < iBins; ++iBin)
        {
            if (vecSAH[iBin] < fBestSah)
            {
                fBestSah = vecSAH[iBin];
                iBestBin = iBin;
            }
        }

        m_pLeft = std::make_shared<BvhTmpNode>(m_itBegin, vecBinIterators[iBestBin + 1], vecLeftBoundingBoxes[iBestBin]);
        m_pRight = std::make_shared<BvhTmpNode>(vecBinIterators[iBestBin + 1], m_itEnd, vecRightBoundingBoxes[iBestBin]);
        m_bIsLeaf = false;
        m_pLeft->Construct();
        m_pRight->Construct();
    }

    std::vector<BvhNode> Serialize() const
    {
        std::vector<BvhNode> vecResult{};
        std::queue<BvhTmpNode const *> queueTodo{};
        queueTodo.push(this);
        while (!queueTodo.empty())
        {
            BvhTmpNode const *pNode = queueTodo.front();
            queueTodo.pop();
            if (!pNode->m_bIsLeaf)
            {
                queueTodo.push(pNode->Left());
                queueTodo.push(pNode->Right());
            }

            vecResult.push_back(pNode->SerializeSingle());
        }
        return vecResult;
    }

private:
    BvhNode SerializeSingle() const
    {
        BvhNode oNode{};
        oNode.m_oBoundingBox = m_oBoundingBox;
        oNode.m_bIsLeaf = m_bIsLeaf;
        return oNode;
    }

    VectorMemory<BoundingBox>::iterator const m_itBegin;
    VectorMemory<BoundingBox>::iterator const m_itEnd;
    std::shared_ptr<BvhTmpNode> m_pLeft;
    std::shared_ptr<BvhTmpNode> m_pRight;
    BoundingBox m_oBoundingBox;
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

    BvhTmpNode oBvhBuilding
    {
        std::begin(m_pScene->m_vecBoundingBoxes) + iBegin,
        std::begin(m_pScene->m_vecBoundingBoxes) + iEnd,
        oSceneBoundingBox
    };
    oBvhBuilding.Construct();
    auto vecBvhNodes = oBvhBuilding.Serialize();
    oTriangleObjDesc.m_iBvhStart = m_vecBvh.size();
    std::move(std::begin(vecBvhNodes), std::end(vecBvhNodes), std::back_inserter(m_vecBvh));
    Assert(false, "BvhManager::AddBvh() => sort triangles (e.g. points + normals), or create a link structure to sort instead!");
}

void BvhManager::Synchronize()
{
    m_vecBvh.Synchronize();
}

BvhNode *BvhManager::CudaPointer()
{
    return m_vecBvh.CudaPointer();
}

} // namespace bvh
} // namespace rtrt

