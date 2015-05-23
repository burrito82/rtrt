#ifndef RTRT_ASSERT_H
#define RTRT_ASSERT_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "LibraryConfig.h"

#include <stdexcept>
#include <string>
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

class RTRTAPI RtrtException : public std::runtime_error
{
public:
    explicit RtrtException(char const *pError):
        std::runtime_error{pError}
    {
    }

    explicit RtrtException(std::string const &strError):
        std::runtime_error(strError)
    {
    }
private:
};

static void Assert(bool bAssert, char const *pError = nullptr)
{
    if (!bAssert)
    {
        throw RtrtException(pError);
    }
}

static void Assert(bool bAssert, std::string const &strError)
{
    if (!bAssert)
    {
        throw RtrtException(strError);
    }
}

} // namespace rtrt


#endif // ! RTRT_ASSERT_H

