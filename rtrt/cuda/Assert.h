#ifndef RTRT_CUDA_ASSERT_H
#define RTRT_CUDA_ASSERT_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include <cuda_runtime.h>

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

class CudaException : public std::runtime_error
{
public:
    CudaException(cudaError_t iError):
        std::runtime_error{"Unspecified error"},
        m_iError{iError}
    {
    }

    explicit CudaException(cudaError_t iError, char const *pError):
        std::runtime_error{pError},
        m_iError{iError}
    {
    }

    explicit CudaException(cudaError_t iError, std::string const &strError):
        std::runtime_error(strError),
        m_iError{iError}
    {
    }
    cudaError_t GetErrorCode() const
    {
        return m_iError;
    }
private:
    cudaError_t m_iError;
};

namespace cuda
{

static void Checked(cudaError_t iError, char const *pError = nullptr)
{
    if (iError != cudaSuccess
        && iError != cudaErrorCudartUnloading) // ignore errors when closing the library
    {
        char const *pErrorString = cudaGetErrorString(iError);
        std::string const strErrMsg = std::string(pErrorString)
            + " (" + std::to_string(iError) + ")"
            + (pError ? std::string("\n\tHint: ") + pError : "");

        throw CudaException(iError, strErrMsg.c_str());
    }
}
static void KernelCheck()
{
    Checked(cudaDeviceSynchronize(), "rtrt::cuda::KernelCheck");
}

} // namespace cuda

} // namespace rtrt


#endif // ! RTRT_CUDA_ASSERT_H

