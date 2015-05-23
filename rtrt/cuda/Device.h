#ifndef RTRT_CUDA_DEVICE_H
#define RTRT_CUDA_DEVICE_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Assert.h"
#include "../Assert.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>
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

class Devices
{
public:
    static Devices &GetInstance()
    {
        static Devices S_oInstance;
        return S_oInstance;
    }

    cudaDeviceProp const &operator[](size_t iDevice) const
    {
        return m_vecProperties[iDevice];
    }

    size_t size() const
    {
        return m_vecProperties.size();
    }

    cudaDeviceProp const &Current() const
    {
        return m_vecProperties[m_iCurrentDevice];
    }

private:
    Devices():
        m_vecProperties{},
        m_iCurrentDevice{}
    {
        int iNumberOfDevices;
        Checked(cudaGetDeviceCount(&iNumberOfDevices));
        Assert(iNumberOfDevices > 0u, "No cuda device found!");

        for (int iDevice = 0u; iDevice < iNumberOfDevices; ++iDevice)
        {
            cudaSetDevice(iDevice);
            cudaDeviceProp oProperty{};
            cudaGetDeviceProperties(&oProperty, iDevice);
            m_vecProperties.push_back(oProperty);
        }

        cudaSetDevice(m_iCurrentDevice);
    }

    int m_iCurrentDevice;
    std::vector<cudaDeviceProp> m_vecProperties;
};

} // namespace cuda
} // namespace rtrt


#endif // ! RTRT_CUDA_DEVICE_H

