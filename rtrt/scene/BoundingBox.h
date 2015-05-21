#ifndef RTRT_SCENE_BOUNDINGBOX_H
#define RTRT_SCENE_BOUNDINGBOX_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "../math/Point.h"
#include "../LibraryConfig.h"

#include <algorithm>
#include <limits>
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

struct RTRTAPI BoundingBox
{
    Point min, max;

    BoundingBox():
        min{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()},
        max{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()}
    {

    }
    BoundingBox(Point const &min_, Point const &max_):
        min{min_},
        max{max_}
    {

    }
    BoundingBox(BoundingBox const &lhs, BoundingBox const &rhs):
        min{lhs.min},
        max{lhs.max}
    {
        this->Grow(rhs);
    }

    bool Contains(Point const &p) const
    {
        return (max.x >= p.x) && (p.x >= min.x)
            && (max.y >= p.y) && (p.y >= min.y)
            && (max.z >= p.z) && (p.z >= min.z);
    }

    Point Center() const
    {
        return min + 0.5f * (max - min);
    }

    float HalfSurfaceArea() const
    {
        Vector d = max - min;
        return d.x * d.y + d.x * d.z + d.y * d.z;
    }

    float SurfaceArea() const
    {
        return 2.0f * HalfSurfaceArea();
    }

    float Volume() const
    {
        Vector d = max - min;
        return d.x * d.y * d.z;
    }

    size_t MaximumExtent() const
    {
        Vector d = max - min;
        if ((d.x > d.y) && (d.x > d.z))
        {
            return 0;
        }
        else
        {
            if (d.y > d.z)
            {
                return 1;
            }
            else
            {
                return 2;
            }
        }
    }

    BoundingBox &Grow(BoundingBox const &rhs)
    {
        min.x = std::min(min.x, rhs.min.x);
        min.y = std::min(min.y, rhs.min.y);
        min.z = std::min(min.z, rhs.min.z);
        max.x = std::max(max.x, rhs.max.x);
        max.y = std::max(max.y, rhs.max.y);
        max.z = std::max(max.z, rhs.max.z);
        return *this;
    }
};

inline BoundingBox &Grow(BoundingBox &lhs, BoundingBox const &rhs)
{
    return lhs.Grow(rhs);
}

inline BoundingBox const Union(BoundingBox lhs, BoundingBox const &rhs)
{
    return Grow(lhs, rhs);
}

inline bool Overlap(BoundingBox const &lhs, BoundingBox const &rhs)
{
    return (lhs.max.x >= rhs.min.x) && (rhs.max.x >= lhs.min.x)
        && (lhs.max.y >= rhs.min.y) && (rhs.max.y >= lhs.min.y)
        && (lhs.max.z >= rhs.min.z) && (rhs.max.z >= lhs.min.z);
}

inline bool Contains(BoundingBox const &bb, Point const &p)
{
    return bb.Contains(p);
}

} // namespace rtrt

#endif // ! RTRT_SCENE_BOUNDINGBOX_H
