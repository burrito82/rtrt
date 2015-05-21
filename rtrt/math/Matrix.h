#ifndef RTRT_MATH_MATRIX_H
#define RTRT_MATH_MATRIX_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../cuda/Float4.cuh"
#include "../LibraryConfig.h"

#include <array>
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
    Matrix(Float4 const &v0, Float4 const &v1, Float4 const &v2, Float4 const &v3):
        m_aRows{}
    {
        m_aRows = {v0, v1, v2, v3};
    }

    Matrix(Float4 const &v0, Float4 const &v1, Float4 const &v2):
        Matrix{v0, v1, v2, Float4{0.0f, 0.0f, 0.0f, 1.0f}}
    {

    }

    explicit Matrix(float a, float b, float c, float d = 1.0f):
        Matrix
        {
            Float4{a, 0.0f, 0.0f, 0.0f},
            Float4{0.0f, b, 0.0f, 0.0f},
            Float4{0.0f, 0.0f, c, 0.0f},
            Float4{0.0f, 0.0f, 0.0f, d}
        }
    {

    }

    Matrix():
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
    Float4 &operator[](size_t index)
    {
        return m_aRows[index];
    }

    Float4 const operator[](size_t index) const
    {
        return const_cast<Matrix *>(this)->operator[](index);
    }

    Matrix const operator+(Matrix const &mat) const
    {
        return
        {
            m_aRows[0] + mat[0],
            m_aRows[1] + mat[1],
            m_aRows[2] + mat[2],
            m_aRows[3] + mat[3]
        };
    }

    Matrix &operator+=(Matrix const &mat)
    {
        m_aRows[0] += mat[0];
        m_aRows[1] += mat[1];
        m_aRows[2] += mat[2];
        m_aRows[3] += mat[3];
        return *this;
    }

    Matrix const operator-(Matrix const &mat) const
    {
        return
        {
            m_aRows[0] - mat[0],
            m_aRows[1] - mat[1],
            m_aRows[2] - mat[2],
            m_aRows[3] - mat[3]
        };
    }

    Matrix &operator-=(Matrix const &mat)
    {
        m_aRows[0] -= mat[0];
        m_aRows[1] -= mat[1];
        m_aRows[2] -= mat[2];
        m_aRows[3] -= mat[3];
        return *this;
    }

    Matrix const Transposed() const;
    static Matrix const Scale(float fScale);
    static Matrix const Translation(Float4 const &v);
    static Matrix const Rotation(Float4 const &n, float fRadians);

private:
    std::array<Float4, 4> m_aRows;
};

RTRTAPI Float4 const operator*(Matrix const &mat, Float4 const &v);
RTRTAPI Matrix const operator*(Matrix const &lhs, Matrix const &rhs);
RTRTAPI Matrix const operator*(float f, Matrix const &mat);

} // namespace rtrt

#endif // ! RTRT_MATH_MATRIX_H
