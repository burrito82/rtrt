#ifndef RTRT_ALIGN_H
#define RTRT_ALIGN_H

// see http://stackoverflow.com/a/12654801
#if defined(_MSC_VER)
#define ALIGNED_TYPE(t,n,i) __declspec(align(i)) t n
#else
#if defined(__GNUC__)
#define ALIGNED_TYPE(t,n,i) t __attribute__ ((aligned(i))) n
#else
#define t n
#endif
#endif

#endif // ! RTRT_ALIGN_H

