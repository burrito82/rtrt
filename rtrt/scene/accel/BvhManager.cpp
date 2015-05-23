#include "BvhManager.h"
/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../Scene.h"
#include "../../Assert.h"

#include <algorithm>
#include <iterator>
#include <queue>
#include <tuple>
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
    BvhTmpNode(VectorMemory<BvhBoundingBox>::iterator const itAbsoluteBegin,
               VectorMemory<BvhBoundingBox>::iterator const itBegin,
               VectorMemory<BvhBoundingBox>::iterator const itEnd,
               BoundingBox const &rBoundingBox):
        m_itAbsoluteBegin{itAbsoluteBegin},
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

    void Construct(int iDepth = 0)
    {
        size_t const iNumberOfTriangles = std::distance(m_itBegin, m_itEnd);
        if (iNumberOfTriangles <= 4u || iDepth > 15)
        {
            return;
        }
        size_t const iMaxBins = 64u;
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

        m_pLeft = std::make_shared<BvhTmpNode>(m_itAbsoluteBegin, m_itBegin, vecBinIterators[iBestBin + 1], vecLeftBoundingBoxes[iBestBin]);
        m_pRight = std::make_shared<BvhTmpNode>(m_itAbsoluteBegin, vecBinIterators[iBestBin + 1], m_itEnd, vecRightBoundingBoxes[iBestBin]);
        m_bIsLeaf = false;
        m_pLeft->Construct(iDepth + 1);
        m_pRight->Construct(iDepth + 1);
    }

    std::vector<BvhNode> Serialize() const
    {
        std::vector<BvhNode> vecResult(MaxSize());
        std::queue<std::tuple<size_t, BvhTmpNode const *>> queueTodo{};
        queueTodo.push(std::make_tuple(0u, this));
        while (!queueTodo.empty())
        {
            size_t iIndex;
            BvhTmpNode const *pNode;
            std::tie(iIndex, pNode) = queueTodo.front();
            queueTodo.pop();
            if (!pNode->m_bIsLeaf)
            {
                queueTodo.push(std::make_tuple(2u * iIndex + 1u, pNode->Left()));
                queueTodo.push(std::make_tuple(2u * iIndex + 2u, pNode->Right()));
            }

            auto oSerialized = pNode->SerializeSingle();
            vecResult[iIndex] = oSerialized;
        }
        return vecResult;
    }

    size_t MaxSize() const
    {
        if (m_bIsLeaf)
        {
            return 1u;
        }
        return 1u + 2u * std::max(m_pLeft->MaxSize(), m_pRight->MaxSize());
    }

private:
    BvhNode SerializeSingle() const
    {
        BvhNode oNode{};
        oNode.m_oBoundingBox = m_oBoundingBox;
        oNode.m_iTriangleIndex = std::distance(m_itAbsoluteBegin, m_itBegin);
        oNode.m_iNumberOfTriangles = std::distance(m_itBegin, m_itEnd);
        oNode.m_bIsLeaf = m_bIsLeaf;
        return oNode;
    }

    VectorMemory<BvhBoundingBox>::iterator const m_itAbsoluteBegin;
    VectorMemory<BvhBoundingBox>::iterator const m_itBegin;
    VectorMemory<BvhBoundingBox>::iterator const m_itEnd;
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
std::vector<bvh::BvhBoundingBox> BvhManager::AddBvh(cuda::TriangleObjectDesc &rTriangleObjDesc)
{
    std::vector<bvh::BvhBoundingBox> vecBoundingBoxes{};
    vecBoundingBoxes.reserve(rTriangleObjDesc.m_iNumberOfTriangles);
    auto iBegin = rTriangleObjDesc.m_iStartIndex;
    auto iEnd = iBegin + rTriangleObjDesc.m_iNumberOfTriangles;
    size_t iWithoutOffset = 0u;
    for (size_t iTriangle = iBegin; iTriangle != iEnd; ++iTriangle)
    {
        cuda::TrianglePoints oTriangle = m_pScene->GetTrianglePoints(iTriangle);
        bvh::BvhBoundingBox oBoundingBox{};
        oBoundingBox.Grow(BoundingBox{thrust::get<0>(oTriangle), thrust::get<0>(oTriangle)});
        oBoundingBox.Grow(BoundingBox{thrust::get<1>(oTriangle), thrust::get<1>(oTriangle)});
        oBoundingBox.Grow(BoundingBox{thrust::get<2>(oTriangle), thrust::get<2>(oTriangle)});
        oBoundingBox.m_iTriangleIndex = iWithoutOffset++;
        oBoundingBox.m_iNumberOfTriangles = 1u;
        vecBoundingBoxes.push_back(oBoundingBox);
    }

    BoundingBox oSceneBoundingBox{};
    for (auto const &bb : vecBoundingBoxes)
    {
        oSceneBoundingBox.Grow(bb);
    }

    BvhTmpNode oBvhBuilding
    {
        std::begin(vecBoundingBoxes),
        std::begin(vecBoundingBoxes),
        std::end(vecBoundingBoxes),
        oSceneBoundingBox
    };
    oBvhBuilding.Construct();
    auto vecBvhNodes = oBvhBuilding.Serialize();
    rTriangleObjDesc.m_iBvhStart = m_vecBvh.size();
    std::move(std::begin(vecBvhNodes), std::end(vecBvhNodes), std::back_inserter(m_vecBvh));
    return vecBoundingBoxes;
}

void BvhManager::Synchronize()
{
    m_vecBvh.Synchronize();
}

BvhNode *BvhManager::CudaPointer()
{
    return m_vecBvh.CudaPointer();
}

BvhNode *BvhManager::data()
{
    return m_vecBvh.data();
}

} // namespace bvh
} // namespace rtrt

