#ifndef RTRT_SCENE_RAYTRIANGLEINTERSECTION_CUH
#define RTRT_SCENE_RAYTRIANGLEINTERSECTION_CUH

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Ray.cuh"
#include "Triangle.h"
#include "../math/Vector.h"

#include <cuda.h>
#include <math.h>
/*============================================================================*/
/* DEFINES                                                                    */
/*============================================================================*/

/*============================================================================*/
/* FORWARD DECLARATIONS                                                       */
/*============================================================================*/

namespace rtrt
{
namespace cuda
{
/*============================================================================*/
/* STRUCT DEFINITIONS                                                         */
/*============================================================================*/

/*============================================================================*/
/* CLASS DEFINITIONS                                                          */
/*============================================================================*/

__device__ __host__ __inline__
float IntersectTriangleWoop(Ray const &rRay, TrianglePoints const &rTriangle)
{
    // Watertight Ray/Triangle Intersection by Woop et al.
    // http://jcgt.org/published/0002/01/05/paper.pdf
    Normal const &dir = rRay.direction;
    int3 k{};
    // max_dim
    k.z = 0;
    if (abs(dir[1]) > abs(dir[k.z])) k.z = 1;
    if (abs(dir[2]) > abs(dir[k.z])) k.z = 2;
    k.x = k.z + 1; if (k.x == 3) k.x = 0;
    k.y = k.x + 1; if (k.y == 3) k.y = 0;

    // preserve winding direction
    if (dir[k.z] < 0.0f)
    {
        auto t = k.x;
        k.x = k.y;
        k.y = t;
    }

    // shear constants
    Vector const S
    {
        dir[k.x] / dir[k.z],
        dir[k.y] / dir[k.z],
        1.0f / dir[k.z]
    };

    // vert rel to ori
    Float4 const A = static_cast<Float4 const &>(thrust::get<0>(rTriangle)) - static_cast<Float4 const &>(rRay.origin);
    Float4 const B = static_cast<Float4 const &>(thrust::get<1>(rTriangle)) - static_cast<Float4 const &>(rRay.origin);
    Float4 const C = static_cast<Float4 const &>(thrust::get<2>(rTriangle)) - static_cast<Float4 const &>(rRay.origin);

    // shear and scale
    float const Ax = A[k.x] - S.x * A[k.z];
    float const Ay = A[k.y] - S.y * A[k.z];
    float const Bx = B[k.x] - S.x * B[k.z];
    float const By = B[k.y] - S.y * B[k.z];
    float const Cx = C[k.x] - S.x * C[k.z];
    float const Cy = C[k.y] - S.y * C[k.z];

    // scaled bary coords
    float U = Cx * By - Cy * Bx;
    float V = Ax * Cy - Ay * Cx;
    float W = Bx * Ay - By * Ax;

    // fallback edge testing
    if (U == 0.0f || V == 0.0f || W == 0.0f)
    {
        double CxBy = static_cast<double>(Cx) * static_cast<double>(By);
        double CyBx = static_cast<double>(Cy) * static_cast<double>(Bx);
        U = static_cast<float>(CxBy - CyBx);

        double AxCy = static_cast<double>(Ax) * static_cast<double>(Cy);
        double AyCx = static_cast<double>(Ay) * static_cast<double>(Cx);
        V = static_cast<float>(AxCy - AyCx);

        double BxAy = static_cast<double>(Bx) * static_cast<double>(Ay);
        double ByAx = static_cast<double>(By) * static_cast<double>(Ax);
        W = static_cast<float>(BxAy - ByAx);
    }

    // edge tests
    if ((U < 0.0f || V < 0.0f || W < 0.0f)
        && (U > 0.0f || V > 0.0f || W > 0.0f))
    {
        return -1.0f;
    }

    // determinant
    float const det = U + V + W;
    if (det == 0.0f)
    {
        return -1.0f;
    }

    // scaled z, calc hit distance
    float const Az = S.z * A[k.z];
    float const Bz = S.z * B[k.z];
    float const Cz = S.z * C[k.z];
    float const T = U * Az + V * Bz + W * Cz;

    /*int det_sign = sign_mask(det);
    if (((T ^ det_sign) < 0.0f)
    || (T ^ det_sign) > hit.t * (det ^ det_sign))
    {
    return -1;
    }*/

    // normalize U, V, W, T
    float const rcpDet = 1.0f / det;
    return T * rcpDet;
}

} // namespace cuda
} // namespace rtrt

#endif // ! RTRT_SCENE_RAYTRIANGLEINTERSECTION_CUH

