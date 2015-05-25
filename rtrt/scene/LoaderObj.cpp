#include "LoaderObj.h"

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../math/Normal.h"
#include "../math/Point.h"

#include <fstream>
#include <sstream>
#include <vector>
/*============================================================================*/
/* MACROS AND DEFINES, CONSTANTS AND STATICS                                  */
/*============================================================================*/
namespace rtrt
{
namespace Loader
{
/*============================================================================*/
/* LOCAL VARS AND FUNCS                                                       */
/*============================================================================*/

/*============================================================================*/
/* CONSTRUCTORS / DESTRUCTOR                                                  */
/*============================================================================*/

/*============================================================================*/
/* IMPLEMENTATION                                                             */
/*============================================================================*/

TriangleGeometry const LoadObj(std::string const &strFilepath)
{
    TriangleGeometry oTriangleObject{};
    std::ifstream oFileHandle{strFilepath};
    std::string strCurLine{};
    std::vector<Point> vecPoints{};
    std::vector<Normal> vecNormals{};

    bool bObjContainsNormals = false;

    while (std::getline(oFileHandle, strCurLine))
    {
        std::istringstream ss{strCurLine};
        std::string strFirstWord;
        ss >> strFirstWord;
        switch (strFirstWord[0])
        {
        case '#':
            continue;
        case 'v':
        {
            if (strFirstWord == "v")
            {
                Point v{};
                ss >> v.x >> v.y >> v.z;
                vecPoints.push_back(v);
                break;
            }
            if (strFirstWord == "vn")
            {
                bObjContainsNormals = true;
                Normal n{};
                ss >> n.x >> n.y >> n.z;
                vecNormals.push_back(n);
                break;
            }
        }
        case 'f':
        {
            std::string aStrIndizes[3]{};
            ss >> aStrIndizes[0] >> aStrIndizes[1] >> aStrIndizes[2];

            for (size_t i = 0; i < 3; ++i)
            {
                std::string strVertex = aStrIndizes[i].substr(0u, aStrIndizes[i].find_first_of('/'));;
                if (strVertex.length() > 0u)
                {
                    int iVertexIndex = std::stoi(strVertex) - 1;
                    oTriangleObject.m_vecPoints.push_back(vecPoints[iVertexIndex]);
                }

                if (bObjContainsNormals)
                {
                    std::string strNormal = aStrIndizes[i].substr(aStrIndizes[i].find_last_of('/') + 1u, aStrIndizes[i].size());
                    if (strNormal.length() > 0u)
                    {
                        int iNormalIndex = std::stoi(strNormal) - 1;
                        oTriangleObject.m_vecNormals.push_back(vecNormals[iNormalIndex]);
                    }
                }
            }
            break;
        }
        default:
        {
            break;
        }
        } // ! switch
    }

    return oTriangleObject;
}

} // namespace Loader
} // namespace rtrt

