#ifndef RTRT_SCENE_TRIANGLEOBJECT_H
#define RTRT_SCENE_TRIANGLEOBJECT_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../cuda/VectorMemory.h"
#include "../math/Normal.h"
#include "../math/Point.h"
#include "../math/Vector.h"
/*============================================================================*/
/* DEFINES                                                                    */
/*============================================================================*/

/*============================================================================*/
/* FORWARD DECLARATIONS                                                       */
/*============================================================================*/

namespace rtrt
{
/*============================================================================*/
/* STRUCT DEFINITIONS                                                         */
/*============================================================================*/

/*============================================================================*/
/* CLASS DEFINITIONS                                                          */
/*============================================================================*/

/**
 * @param
 * @return
 * @see
 * @todo
 * @bug
 * @deprecated
 */
class TriangleObject
{
public:
    TriangleObject()
    {
    }
protected:
private:
    VectorMemory<Vector> m_vecVertices;
    VectorMemory<Normal> m_vecNormals;
};

} // namespace rtrt

#endif // ! RTRT_SCENE_TRIANGLEOBJECT_H

