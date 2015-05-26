#include "TypeVerifier.h"
/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Assert.h"
#include "Float4.cuh"
#include "TypeVerifier.cuh"

#include "../Assert.h"
#include "../math/Normal.h"
#include "../math/Point.h"
#include "../math/Vector.h"
#include "../scene/HitPoint.h"
#include "../scene/Material.h"
#include "../scene/Ray.cuh"

#include <iostream>
#include <string>
#include <typeinfo>
/*============================================================================*/
/* MACROS AND DEFINES, CONSTANTS AND STATICS                                  */
/*============================================================================*/
namespace rtrt
{
namespace cuda
{
/*============================================================================*/
/* LOCAL VARS AND FUNCS                                                       */
/*============================================================================*/
template<typename T>
void Verify()
{
    if (sizeof(Ray) != GetTypeSize(Ray{}))
    {
        std::string strError = "TypeVerifier::VerifySize<";
        strError += typeid(T).name();
        strError += ">() failed, ";
        strError += std::to_string(sizeof(Ray)) + " != ";
        strError += std::to_string(GetTypeSize(Ray{}));
        std::cerr << strError << std::endl;
        throw RtrtException(strError);
    }
}
/*============================================================================*/
/* CONSTRUCTORS / DESTRUCTOR                                                  */
/*============================================================================*/

/*============================================================================*/
/* IMPLEMENTATION                                                             */
/*============================================================================*/

bool TypeVerifier::VerifySize()
{
    Verify<HitPoint>();
    Verify<Ray>();
    Verify<Float4>();
    Verify<Material>();
    Verify<Normal>();
    Verify<Point>();
    Verify<Vector>();
    return true;
}

} // namespace cuda
} // namespace rtrt

