#ifndef RTRT_CUDA_TYPEVERIFIER_CUH
#define RTRT_CUDA_TYPEVERIFIER_CUH

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/

/*============================================================================*/
/* DEFINES                                                                    */
/*============================================================================*/

/*============================================================================*/
/* FORWARD DECLARATIONS                                                       */
/*============================================================================*/
namespace rtrt
{
struct Ray;
struct Float4;
struct Material;
struct Normal;
struct Point;
struct Vector;
namespace cuda
{
struct HitPoint;
/*============================================================================*/
/* STRUCT DEFINITIONS                                                         */
/*============================================================================*/

size_t GetTypeSize(HitPoint const &);
size_t GetTypeSize(Ray const &);
size_t GetTypeSize(Float4 const &);
size_t GetTypeSize(Material const &);
size_t GetTypeSize(Normal const &);
size_t GetTypeSize(Point const &);
size_t GetTypeSize(Vector const &);

} // namespace cuda
} // namespace rtrt


#endif // ! RTRT_CUDA_TYPEVERIFIER_CUH

