#ifndef RTRT_MATH_MATRIX_H
#define RTRT_MATH_MATRIX_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../cuda/Defines.h"
#include "../cuda/Float4.cuh"

#include <cmath>
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

struct RTRTAPI Matrix
{
    RTRTDH Matrix(Float4 const &v0, Float4 const &v1, Float4 const &v2, Float4 const &v3):
        m_aRows{}
    {
        m_aRows[0] = v0; 
        m_aRows[1] = v1; 
        m_aRows[2] = v2; 
        m_aRows[3] = v3;
    }

    RTRTDH Matrix(Float4 const &v0, Float4 const &v1, Float4 const &v2):
        Matrix{v0, v1, v2, Float4{0.0f, 0.0f, 0.0f, 1.0f}}
    {

    }

    RTRTDH explicit Matrix(float a, float b, float c, float d = 1.0f):
        Matrix
        {
            Float4{a, 0.0f, 0.0f, 0.0f},
            Float4{0.0f, b, 0.0f, 0.0f},
            Float4{0.0f, 0.0f, c, 0.0f},
            Float4{0.0f, 0.0f, 0.0f, d}
        }
    {

    }

    RTRTDH Matrix():
        Matrix
        {
            Float4{1.0f, 0.0f, 0.0f, 0.0f},
            Float4{0.0f, 1.0f, 0.0f, 0.0f},
            Float4{0.0f, 0.0f, 1.0f, 0.0f},
            Float4{0.0f, 0.0f, 0.0f, 1.0f}
        }
    {

    }

    // index access
    RTRTDHL Float4 &operator[](size_t index)
    {
        return m_aRows[index];
    }

    RTRTDHL Float4 const operator[](size_t index) const
    {
        return const_cast<Matrix *>(this)->operator[](index);
    }

    RTRTDHL Matrix const operator+(Matrix const &mat) const
    {
        return
        {
            m_aRows[0] + mat[0],
            m_aRows[1] + mat[1],
            m_aRows[2] + mat[2],
            m_aRows[3] + mat[3]
        };
    }

    RTRTDHL Matrix &operator+=(Matrix const &mat)
    {
        m_aRows[0] += mat[0];
        m_aRows[1] += mat[1];
        m_aRows[2] += mat[2];
        m_aRows[3] += mat[3];
        return *this;
    }

    RTRTDHL Matrix const operator-(Matrix const &mat) const
    {
        return
        {
            m_aRows[0] - mat[0],
            m_aRows[1] - mat[1],
            m_aRows[2] - mat[2],
            m_aRows[3] - mat[3]
        };
    }

    RTRTDHL Matrix &operator-=(Matrix const &mat)
    {
        m_aRows[0] -= mat[0];
        m_aRows[1] -= mat[1];
        m_aRows[2] -= mat[2];
        m_aRows[3] -= mat[3];
        return *this;
    }

    RTRTDHL bool operator!=(Matrix const &mat) const
    {
        return (m_aRows[0][0] != mat[0][0]
                || m_aRows[0][1] != mat[0][1]
                || m_aRows[0][2] != mat[0][2]
                || m_aRows[0][3] != mat[0][3]
                || m_aRows[1][0] != mat[1][0]
                || m_aRows[1][1] != mat[1][1]
                || m_aRows[1][2] != mat[1][2]
                || m_aRows[1][3] != mat[1][3]
                || m_aRows[2][0] != mat[2][0]
                || m_aRows[2][1] != mat[2][1]
                || m_aRows[2][2] != mat[2][2]
                || m_aRows[2][3] != mat[2][3]
                || m_aRows[3][0] != mat[3][0]
                || m_aRows[3][1] != mat[3][1]
                || m_aRows[3][2] != mat[3][2]
                || m_aRows[3][3] != mat[3][3]);
    }

    RTRTDHL Matrix const Transposed() const;
    RTRTDHL Matrix const Inverted() const;
    RTRTDHL static Matrix const Scale(float fScale);
    RTRTDHL static Matrix const Translation(Float4 const &v);
    RTRTDHL static Matrix const Rotation(Float4 const &n, float fRadians);
    RTRTDHL Matrix ExtractTranslation() const;

private:
    RTRTDHL Matrix const Inverted3x3() const;
    Float4 m_aRows[4];
};

RTRTDHLAPI Float4 const operator*(Matrix const &mat, Float4 const &v);
RTRTDHLAPI Matrix const operator*(Matrix const &lhs, Matrix const &rhs);
RTRTDHLAPI Matrix const operator*(float f, Matrix const &mat);

} // namespace rtrt

#include "Matrix.inl"

#endif // ! RTRT_MATH_MATRIX_H
