#ifndef RTRT_MATH_MATRIX_INL
#define RTRT_MATH_MATRIX_INL
#include "Matrix.h"
/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/

/*============================================================================*/
/* MACROS AND DEFINES, CONSTANTS AND STATICS                                  */
/*============================================================================*/
namespace rtrt
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

RTRTDHL Matrix const Matrix::Transposed() const
{
    Matrix matResult{};
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            matResult[i][j] = m_aRows[j][i];
        }
    }
    return matResult;
}

RTRTDHL Matrix const Matrix::Scale(float fScale)
{
    Matrix matResult{};
    matResult[0][0] = fScale;
    matResult[1][1] = fScale;
    matResult[2][2] = fScale;
    return matResult;
}

RTRTDHL Matrix const Matrix::Translation(Float4 const &v)
{
    Matrix matResult{};
    matResult[3][0] = v[0];
    matResult[3][1] = v[1];
    matResult[3][2] = v[2];
    return matResult;
}

RTRTDHL Matrix const Matrix::Rotation(Float4 const &n, float fRadians)
{
    Matrix const nnT
    {
        n * n[0],
        n * n[1],
        n * n[2],
        n * n[3]
    };

    Matrix const X
    {
        Float4{0.0f, -n.z, n.y, 0.0f},
        Float4{n.z, 0.0f, -n.x, 0.0f},
        Float4{-n.y, n.x, 0.0f, 0.0f},
        Float4{0.0f, 0.0f, 0.0f, 1.0f}
    };

    return std::cos(fRadians) * Matrix{}
        +(1 - std::cos(fRadians)) * nnT
        - std::sin(fRadians) * X;
}

RTRTDHLAPI Float4 const operator*(Matrix const &mat, Float4 const &v)
{
    return
    {
        mat[0][0] * v[0] + mat[1][0] * v[1] + mat[2][0] * v[2] + mat[3][0] * v[3],
        mat[0][1] * v[0] + mat[1][1] * v[1] + mat[2][1] * v[2] + mat[3][1] * v[3],
        mat[0][2] * v[0] + mat[1][2] * v[1] + mat[2][2] * v[2] + mat[3][2] * v[3],
        mat[0][3] * v[0] + mat[1][3] * v[1] + mat[2][3] * v[2] + mat[3][3] * v[3]
    };
}

RTRTDHLAPI Matrix const operator*(Matrix const &lhs, Matrix const &rhs)
{
    return
    {
        lhs * rhs[0],
        lhs * rhs[1],
        lhs * rhs[2],
        lhs * rhs[3]
    };
}

RTRTDHLAPI Matrix const operator*(float f, Matrix const &mat)
{
    return mat * Matrix{f, f, f, f};
}

} // namespace rtrt

#endif // ! RTRT_MATH_MATRIX_INL
