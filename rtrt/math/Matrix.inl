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


RTRTDHL Matrix const Matrix::Inverted() const
{
    Matrix matResult{0.0f, 0.0f, 0.0f};

    auto inv3x3 = Inverted3x3();
    matResult += inv3x3;
    matResult[3] = -(inv3x3 * m_aRows[3]);
    matResult[3][3] = 1.0f;

    return matResult;
}

RTRTDHL Matrix const Matrix::Inverted3x3() const
{
    Matrix matResult{};
    auto const &a = m_aRows;
    float fDet = a[0][0] * a[1][1] * a[2][2]
        + a[0][1] * a[1][2] * a[2][0]
        + a[0][2] * a[1][0] * a[2][1]
        - a[0][0] * a[1][2] * a[2][1]
        - a[0][2] * a[1][1] * a[2][0]
        - a[0][1] * a[1][0] * a[2][2];

    matResult[0][0] = a[1][1] * a[2][2] - a[2][1] * a[1][2];
    matResult[0][1] = a[2][1] * a[0][2] - a[0][1] * a[2][2];
    matResult[0][2] = a[0][1] * a[1][2] - a[1][1] * a[0][2];

    matResult[1][0] = a[2][0] * a[1][2] - a[1][0] * a[2][2];
    matResult[1][1] = a[0][0] * a[2][2] - a[2][0] * a[0][2];
    matResult[1][2] = a[1][0] * a[0][2] - a[0][0] * a[1][2];

    matResult[2][0] = a[1][0] * a[2][1] - a[2][0] * a[1][1];
    matResult[2][1] = a[2][0] * a[0][1] - a[0][0] * a[2][1];
    matResult[2][2] = a[0][0] * a[1][1] - a[1][0] * a[0][1];

    matResult = (1.0f / fDet) * matResult;

    return matResult;
}

RTRTDHL Matrix const Matrix::Scale(float fScale)
{
    return Matrix{fScale, fScale, fScale};
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
        Float4{}
    };

    Matrix const X
    {
        Float4{0.0f, n.z, -n.y, 0.0f},
        Float4{-n.z, 0.0f, n.x, 0.0f},
        Float4{n.y, -n.x, 0.0f, 0.0f},
        Float4{0.0f, 0.0f, 0.0f, 1.0f}
    };

    float cosA = std::cos(fRadians);
    auto matResult = Matrix{cosA, cosA, cosA}
        +(1 - cosA) * nnT
        + std::sin(fRadians) * X;
    matResult[3] = Float4{0.0f, 0.0f, 0.0f, 1.0f};
    return matResult;
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
