#ifndef RTRT_CUDA_VECTORMEMORY_INL
#define RTRT_CUDA_VECTORMEMORY_INL

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include <cuda_runtime.h>

#include <algorithm>
/*============================================================================*/
/* MACROS AND DEFINES, CONSTANTS AND STATICS                                  */
/*============================================================================*/
namespace rtrt
{
/*============================================================================*/
/* LOCAL VARS AND FUNCS                                                       */
/*============================================================================*/
namespace
{
template<WriteDirection WRITEDIRECTION>
struct Synchronizer
{
    template<typename T>
    static void Synchronize(T *pCpu, T *pGpu, size_t count);
};

template<>
template<typename T>
void Synchronizer<CPU_TO_GPU>::Synchronize(T *pCpu, T *pGpu, size_t count)
{
    cuda::Checked(cudaMemcpy(pGpu, pCpu, sizeof(T) * count, cudaMemcpyHostToDevice));
}

template<>
template<typename T>
void Synchronizer<GPU_TO_CPU>::Synchronize(T *pCpu, T *pGpu, size_t count)
{
    cuda::Checked(cudaMemcpy(pCpu, pGpu, sizeof(T) * count, cudaMemcpyDeviceToHost));
}
}
/*============================================================================*/
/* CONSTRUCTORS / DESTRUCTOR                                                  */
/*============================================================================*/

template<typename T, WriteDirection WRITEDIRECTION>
VectorMemory<T, WRITEDIRECTION>::VectorMemory():
    m_vecCpu(),
    m_iCudaSize(0),
    m_iCudaCapacity(0),
    m_pCuda{nullptr}
{
    reserve(m_vecCpu.capacity());
    resize(m_vecCpu.size());
}

template<typename T, WriteDirection WRITEDIRECTION>
VectorMemory<T, WRITEDIRECTION>::VectorMemory(size_t iSize):
    m_vecCpu(iSize),
    m_iCudaSize(0),
    m_iCudaCapacity(0),
    m_pCuda{nullptr}
{
    reserve(m_vecCpu.capacity());
    resize(m_vecCpu.size());
}

template<typename T, WriteDirection WRITEDIRECTION>
VectorMemory<T, WRITEDIRECTION>::VectorMemory(VectorMemory<T, WRITEDIRECTION> const &rhs):
    m_vecCpu(rhs.m_vecCpu),
    m_iCudaSize(0),
    m_iCudaCapacity(0),
    m_pCuda{nullptr}
{
    reserve(m_vecCpu.capacity());
    resize(m_vecCpu.size());
    if (rhs.m_pCuda.get())
    {
        cuda::Checked(cudaMemcpy(m_pCuda.get(), rhs.m_pCuda.get(), sizeof(T) * m_iCudaSize, cudaMemcpyDeviceToDevice));
    }
}

template<typename T, WriteDirection WRITEDIRECTION>
VectorMemory<T, WRITEDIRECTION>::VectorMemory(VectorMemory<T, WRITEDIRECTION> &&rhs):
    m_vecCpu(std::move(rhs.m_vecCpu)),
    m_iCudaSize{rhs.m_iCudaSize},
    m_iCudaCapacity{rhs.m_iCudaCapacity},
    m_pCuda{rhs.m_pCuda}
{
    rhs.m_iCudaSize = 0;
    rhs.m_iCudaCapacity = 0;
    rhs.m_pCuda = nullptr;
}

template<typename T, WriteDirection WRITEDIRECTION>
VectorMemory<T, WRITEDIRECTION>::VectorMemory(Container<T> const &rhs):
    m_vecCpu(rhs.m_vecCpu),
    m_iCudaSize(),
    m_iCudaCapacity(0),
    m_pCuda{nullptr}
{
    reserve(m_vecCpu.capacity());
    resize(m_vecCpu.size());
    Cpu2Gpu();
}

template<typename T, WriteDirection WRITEDIRECTION>
VectorMemory<T, WRITEDIRECTION>::VectorMemory(Container<T> &&rhs):
    m_vecCpu(std::move(rhs)),
    m_iCudaSize(0),
    m_iCudaCapacity(0),
    m_pCuda{nullptr}
{
    reserve(m_vecCpu.capacity());
    resize(m_vecCpu.size());
    Cpu2Gpu();
}

template<typename T, WriteDirection WRITEDIRECTION>
VectorMemory<T, WRITEDIRECTION>::~VectorMemory()
{
}
/*============================================================================*/
/* IMPLEMENTATION                                                             */
/*============================================================================*/
template<typename T, WriteDirection WRITEDIRECTION>
VectorMemory<T, WRITEDIRECTION>::operator std::vector<T>() const
{
    return m_vecCpu;
}

template<typename T, WriteDirection WRITEDIRECTION>
VectorMemory<T, WRITEDIRECTION> &VectorMemory<T, WRITEDIRECTION>::operator=(VectorMemory<T, WRITEDIRECTION> const &rhs)
{
    m_vecCpu = rhs.m_vecCpu;
    reserve(rhs.capacity());
    resize(rhs.size());
    cuda::Checked(cudaMemcpy(m_pCuda.get(), rhs.m_pCuda.get(), sizeof(T) * rhs.m_iCudaSize, cudaMemcpyDeviceToDevice));
    return *this;
}

template<typename T, WriteDirection WRITEDIRECTION>
VectorMemory<T, WRITEDIRECTION> &VectorMemory<T, WRITEDIRECTION>::operator=(VectorMemory<T, WRITEDIRECTION> &&rhs)
{
    swap(rhs);
    rhs.m_iCudaSize = 0;
    return *this;
}

template<typename T, WriteDirection WRITEDIRECTION>
VectorMemory<T, WRITEDIRECTION> &VectorMemory<T, WRITEDIRECTION>::operator=(Container<T> const &rhs)
{
    m_vecCpu = rhs;
    reserve(rhs.capacity());
    resize(rhs.size());
    Cpu2Gpu();
    return *this;
}

template<typename T, WriteDirection WRITEDIRECTION>
VectorMemory<T, WRITEDIRECTION> &VectorMemory<T, WRITEDIRECTION>::operator=(Container<T> &&rhs)
{
    m_vecCpu = std::move(rhs);
    reserve(rhs.capacity());
    resize(rhs.size());
    Cpu2Gpu();
    return *this;
}

template<typename T, WriteDirection WRITEDIRECTION>
VectorMemory<T, WRITEDIRECTION> &VectorMemory<T, WRITEDIRECTION>::operator=(std::initializer_list<T> ilist)
{
    resize(ilist.size());
    std::copy(std::begin(ilist), std::end(ilist), std::begin(m_vecCpu));
    reserve(capacity());
    resize(size());
    Cpu2Gpu();
    return *this;
}

template<typename T, WriteDirection WRITEDIRECTION>
typename VectorMemory<T, WRITEDIRECTION>::reference VectorMemory<T, WRITEDIRECTION>::operator[](size_type index)
{
    return m_vecCpu[index];
}

template<typename T, WriteDirection WRITEDIRECTION>
typename VectorMemory<T, WRITEDIRECTION>::const_reference VectorMemory<T, WRITEDIRECTION>::operator[](size_type index) const
{
    return m_vecCpu[index];
}

template<typename T, WriteDirection WRITEDIRECTION>
T *VectorMemory<T, WRITEDIRECTION>::data()
{
    return m_vecCpu.data();
}

template<typename T, WriteDirection WRITEDIRECTION>
typename VectorMemory<T, WRITEDIRECTION>::iterator VectorMemory<T, WRITEDIRECTION>::begin()
{
    return m_vecCpu.begin();
}

template<typename T, WriteDirection WRITEDIRECTION>
typename VectorMemory<T, WRITEDIRECTION>::const_iterator VectorMemory<T, WRITEDIRECTION>::begin() const
{
    return m_vecCpu.begin();
}

template<typename T, WriteDirection WRITEDIRECTION>
typename VectorMemory<T, WRITEDIRECTION>::iterator VectorMemory<T, WRITEDIRECTION>::end()
{
    return m_vecCpu.end();
}

template<typename T, WriteDirection WRITEDIRECTION>
typename VectorMemory<T, WRITEDIRECTION>::const_iterator VectorMemory<T, WRITEDIRECTION>::end() const
{
    return m_vecCpu.end();
}

template<typename T, WriteDirection WRITEDIRECTION>
bool VectorMemory<T, WRITEDIRECTION>::empty() const
{
    return m_vecCpu.empty();
}

template<typename T, WriteDirection WRITEDIRECTION>
typename VectorMemory<T, WRITEDIRECTION>::size_type VectorMemory<T, WRITEDIRECTION>::size() const
{
    return m_vecCpu.size();
}

template<typename T, WriteDirection WRITEDIRECTION>
void VectorMemory<T, WRITEDIRECTION>::reserve(size_type new_cap)
{
    using std::swap;
    m_vecCpu.reserve(new_cap);
    new_cap = m_vecCpu.capacity();
    if (m_iCudaCapacity < new_cap)
    {
        T *pCuda = 0;
        cuda::Checked(cudaMalloc(&pCuda, new_cap * sizeof(T)), "cudaMalloc failed!");
        decltype(m_pCuda) pBigger{pCuda};
        if (m_pCuda != nullptr && m_iCudaSize > 0)
        {
            cuda::Checked(cudaMemcpy(pBigger.get(), m_pCuda.get(), sizeof(T) * m_iCudaSize, cudaMemcpyDeviceToDevice));
        }
        swap(m_pCuda, pBigger);
        m_iCudaCapacity = new_cap;
    }
}

template<typename T, WriteDirection WRITEDIRECTION>
typename VectorMemory<T, WRITEDIRECTION>::size_type VectorMemory<T, WRITEDIRECTION>::capacity() const
{
    return m_vecCpu.capacity();
}


template<typename T, WriteDirection WRITEDIRECTION>
void VectorMemory<T, WRITEDIRECTION>::clear()
{
    m_vecCpu.clear();
}

template<typename T, WriteDirection WRITEDIRECTION>
typename VectorMemory<T, WRITEDIRECTION>::iterator VectorMemory<T, WRITEDIRECTION>::insert(const_iterator pos, T const &value)
{
    return m_vecCpu.insert(pos, value);
}

template<typename T, WriteDirection WRITEDIRECTION>
typename VectorMemory<T, WRITEDIRECTION>::iterator VectorMemory<T, WRITEDIRECTION>::erase(const_iterator pos)
{
    return m_vecCpu.erase(pos);
}

template<typename T, WriteDirection WRITEDIRECTION>
void VectorMemory<T, WRITEDIRECTION>::push_back(T const &value)
{
    m_vecCpu.push_back(value);
}

template<typename T, WriteDirection WRITEDIRECTION>
void VectorMemory<T, WRITEDIRECTION>::push_back(T &&value)
{
    m_vecCpu.push_back(std::forward<T>(value));
}

template<typename T, WriteDirection WRITEDIRECTION>
void VectorMemory<T, WRITEDIRECTION>::resize(size_type count)
{
    if (m_vecCpu.size() < count)
    {
        m_vecCpu.resize(count);
    }

    if (m_iCudaSize < count)
    {
        reserve(m_vecCpu.capacity());
        m_iCudaSize = count;
    }
}

template<typename T, WriteDirection WRITEDIRECTION>
void VectorMemory<T, WRITEDIRECTION>::swap(VectorMemory<T, WRITEDIRECTION> &rhs)
{
    using std::swap;
    swap(m_vecCpu, rhs.m_vecCpu);
    swap(m_iCudaSize, rhs.m_iCudaSize);
    swap(m_iCudaCapacity, rhs.m_iCudaCapacity);
    swap(m_pCuda, rhs.m_pCuda);
}

template<typename T, WriteDirection WRITEDIRECTION>
typename VectorMemory<T, WRITEDIRECTION>::pointer VectorMemory<T, WRITEDIRECTION>::CudaPointer()
{
    return m_pCuda.get();
}

template<typename T, WriteDirection WRITEDIRECTION>
typename VectorMemory<T, WRITEDIRECTION>::const_pointer VectorMemory<T, WRITEDIRECTION>::CudaPointer() const
{
    return m_pCuda.get();
}

template<typename T, WriteDirection WRITEDIRECTION>
void VectorMemory<T, WRITEDIRECTION>::Synchronize()
{
    reserve(WRITEDIRECTION == CPU_TO_GPU ? m_vecCpu.capacity() : m_iCudaCapacity);
    resize(WRITEDIRECTION == CPU_TO_GPU ? m_vecCpu.size() : m_iCudaSize);
    Synchronizer<WRITEDIRECTION>::Synchronize(m_vecCpu.data(), m_pCuda.get(), size());
}

template<typename T, WriteDirection WRITEDIRECTION>
void VectorMemory<T, WRITEDIRECTION>::Cpu2Gpu()
{
    reserve(m_vecCpu.capacity());
    resize(m_vecCpu.size());
    Synchronizer<CPU_TO_GPU>::Synchronize(m_vecCpu.data(), m_pCuda.get(), size());
}

template<typename T, WriteDirection WRITEDIRECTION>
void VectorMemory<T, WRITEDIRECTION>::Gpu2Cpu()
{
    reserve(m_iCudaCapacity);
    resize(m_iCudaSize);
    Synchronizer<GPU_TO_CPU>::Synchronize(m_vecCpu.data(), m_pCuda.get(), size());
}

} // namespace rtrt

#endif // ! RTRT_CUDA_VECTORMEMORY_INL
