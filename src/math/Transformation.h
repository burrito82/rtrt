#ifndef RTRT_MATH_TRANSFORMATION_H
#define RTRT_MATH_TRANSFORMATION_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Vector.h"
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
    Matrix():
        m_aRows{
            Vector{1.0f, 0.0f, 0.0f, 0.0f}, 
            Vector{0.0f, 1.0f, 0.0f, 0.0f}, 
            Vector{0.0f, 0.0f, 1.0f, 0.0f}, 
            Vector{0.0f, 0.0f, 0.0f, 1.0f}
        }
    {

    }

    // index access
    Vector &operator[](size_t index)
    {
        return m_aRows[index];
    }

    Vector operator[](size_t index) const
    {
        return const_cast<Matrix *>(this)->operator[](index);
    }

private:
    std::array<Vector, 4> m_aRows;
};

} // namespace rtrt

#endif // ! RTRT_MATH_TRANSFORMATION_H
