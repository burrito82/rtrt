#ifndef RTRT_RAY_H
#define RTRT_RAY_H

#include "../LibraryConfig.h"
#include "../math/Normal.h"
#include "../math/Vector.h"

namespace rtrt
{

struct RTRTAPI Ray
{
    Vector origin;
    Normal direction;
};

}

#endif // ! RTRT_RAY_H
