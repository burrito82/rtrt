#ifndef RTRT_VECTOR_H
#define RTRT_VECTOR_H

#include <vector_types.h>

namespace rtrt
{

struct RTRTAPI Vector : public float4
{

};

Vector const operator+(Vector const &lhs, Vector const &rhs);
Vector const operator-(Vector const &lhs, Vector const &rhs);

Vector const operator*(float lhs, Vector const &rhs);
Vector const operator*(Vector const &lhs, float rhs);
Vector const operator/(Vector const &lhs, float rhs);

float Dot(Vector const &lhs, Vector const &rhs);
float Length(Vector const &v);

}

#endif // ! RTRT_VECTOR_H
