#ifndef RTRT_CUDA_VECTORMEMORY_H
#define RTRT_CUDA_VECTORMEMORY_H

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include "Assert.h"
#include <cuda.h>
#include <initializer_list>
#include <memory>
#include <vector>
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
enum WriteDirection
{
    CPU_TO_GPU,
    GPU_TO_CPU
};

template<typename T, WriteDirection WRITEDIRECTION = CPU_TO_GPU>
class VectorMemory
{
public:
    template<typename CT> using Container = std::vector<CT>;
    using value_type = T;
    using size_type = typename Container<T>::size_type;
    using difference_type = typename Container<T>::difference_type;
    using reference = typename Container<T>::reference;
    using const_reference = typename Container<T>::const_reference;
    using pointer = typename Container<T>::pointer;
    using const_pointer = typename Container<T>::const_pointer;
    using iterator = typename Container<T>::iterator;
    using const_iterator = typename Container<T>::const_iterator;
    using reverse_iterator = typename Container<T>::reverse_iterator;
    using const_reverse_iterator = typename Container<T>::const_reverse_iterator;

    VectorMemory();
    explicit VectorMemory(size_t iSize);
    explicit VectorMemory(VectorMemory<T, WRITEDIRECTION> const &rhs);
    explicit VectorMemory(VectorMemory<T, WRITEDIRECTION> &&rhs);
    explicit VectorMemory(Container<T> const &rhs);
    explicit VectorMemory(Container<T> &&rhs);
    ~VectorMemory();

    explicit operator std::vector<T>() const;

    VectorMemory<T, WRITEDIRECTION> &operator=(VectorMemory<T, WRITEDIRECTION> const &rhs);
    VectorMemory<T, WRITEDIRECTION> &operator=(VectorMemory<T, WRITEDIRECTION> &&rhs);
    VectorMemory<T, WRITEDIRECTION> &operator=(Container<T> const &rhs);
    VectorMemory<T, WRITEDIRECTION> &operator=(Container<T> &&rhs);
    VectorMemory<T, WRITEDIRECTION> &operator=(std::initializer_list<T> ilist);

    reference operator[](size_type index);
    const_reference operator[](size_type index) const;

    T *data();
    iterator begin();
    const_iterator begin() const;
    iterator end();
    const_iterator end() const;

    bool empty() const;
    size_type size() const;
    void reserve(size_type new_cap);
    size_type capacity() const;

    void clear();
    iterator insert(const_iterator pos, T const &value);
    iterator erase(const_iterator pos);
    void push_back(T const &value);
    void push_back(T &&value);
    void resize(size_type count);
    void swap(VectorMemory<T, WRITEDIRECTION> &rhs);

    pointer CudaPointer();
    void Synchronize();
    void Cpu2Gpu();
    void Gpu2Cpu();

private:
    template<typename CT>
    struct CudaDeleter
    {
        void operator()(CT *ptr) const
        {
            cuda::Checked(cudaFree(ptr));
        }
    };

    Container<T> m_vecCpu;
    size_type m_iCudaSize;
    size_type m_iCudaCapacity;
    std::unique_ptr<T, CudaDeleter<T>> m_pCuda;
};

} // namespace rtrt

#include "VectorMemory.inl"

namespace std
{
    template<typename T, rtrt::WriteDirection WDIR>
    void swap(rtrt::VectorMemory<T, WDIR> &lhs, rtrt::VectorMemory<T, WDIR> &rhs)
    {
        lhs.swap(rhs);
    }
}

#endif // ! RTRT_CUDA_VECTORMEMORY_H
