#ifndef RTRT_POINT_H
#define RTRT_POINT_H

#include "Vector.h"

#include <vector_types.h>

namespace rtrt
{

struct RTRTAPI Point : public float4
{

};

Point const operator+(Point const &p, Vector const &v);
Point const operator+(Vector const &v, Point const &p);
Point const operator-(Point const &p, Vector const &v);

Vector const operator-(Point const &lhs, Point const &rhs);

}

#endif // ! RTRT_POINT_H
