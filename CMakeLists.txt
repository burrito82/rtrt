# $Id:$

cmake_minimum_required( VERSION 2.8 )
project( rtrt )

option(USE_VISTA "Use Vista CMake build system" off)
if(USE_VISTA)
	list( APPEND CMAKE_MODULE_PATH "$ENV{VISTA_CMAKE_COMMON}" )
	include( VistaCommon )

include( VistaCommon )
endif(USE_VISTA)

option (RTRT_USE_CUDA "Rtrt with CUDA support" off)
if (RTRT_USE_CUDA)
    add_definitions(-DRTRT_USE_CUDA)
    if(USE_VISTA)
        vista_use_package( CUDA 6.0 REQUIRED FIND_DEPENDENCIES)
    else(USE_VISTA)
        include(GenerateExportHeader)
        find_package(CUDA)
    endif(USE_VISTA)
endif (RTRT_USE_CUDA)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(CMAKE_COMPILER_IS_GNUCC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif(CMAKE_COMPILER_IS_GNUCC)

#CUDA

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE off)
# output usage of register etc.
set( CUDA_VERBOSE_BUILD ON )

# optimization, trade precision for speed
set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-keep;-ftz=true;-prec-div=false;-prec-sqrt=false)

# set nvcc-flags:   ( see nvcc --help )
#   -arch=sm_20:          compile for fermi architecture, see deviceQuery in the CUDA-SDK
set( CUDA_NVCC_FLAGS_ARCH sm_35 CACHE STRING "CUDA compute capability. sm_20 for fermi architecture. See deviceQuery from CUDA SDK." )
if( CUDA_NVCC_FLAGS_ARCH )
	set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=${CUDA_NVCC_FLAGS_ARCH} )
endif( CUDA_NVCC_FLAGS_ARCH )

#   -use_fast_math:       Make use of fast math library.
set( CUDA_NVCC_FLAGS_FASTMATH 1 CACHE BOOL "Make use of fast math library." )
if( CUDA_NVCC_FLAGS_FASTMATH)
	set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-use_fast_math )
endif( CUDA_NVCC_FLAGS_FASTMATH)

#   -maxrregcount=64:     specify max. number of register per block
set( CUDA_NVCC_FLAGS_MAXREG 64 CACHE STRING "specify max. number of register per block." )
if( CUDA_NVCC_FLAGS_MAXREG)
	set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-maxrregcount=${CUDA_NVCC_FLAGS_MAXREG} )
endif( CUDA_NVCC_FLAGS_MAXREG)

# set release nvcc-flags:   ( see nvcc --help )
set( CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE};-O3 )

# set debug nvcc-flags:   ( see nvcc --help )
set( CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};-g;-G )

# print kernel runtimes
set( CUDA_KERNEL_VERBOSE 0 CACHE BOOL "Output kernel compute timings when running.")
if( CUDA_KERNEL_VERBOSE )
	add_definitions( -DKERNEL_VERBOSE )
endif( CUDA_KERNEL_VERBOSE )


include( "./rtrt/_SourceFiles.cmake" )
if (RTRT_USE_CUDA)
CUDA_ADD_LIBRARY( rtrt SHARED ${ProjectSources})
else (RTRT_USE_CUDA)
add_library( rtrt SHARED ${ProjectSources})
endif (RTRT_USE_CUDA)
#END CUDA

#The following line prevent CMake from adding all depencies to other projects that link it from within the same cmake build
set_property( TARGET rtrt PROPERTY LINK_INTERFACE_LIBRARIES "" )
target_link_libraries( rtrt 
)

if(USE_VISTA)
vista_configure_lib( rtrt )
#if( rtrt_COMMON_BUILD )
#    vista_install( rtrt rtrt )
#else( rtrt_COMMON_BUILD )
	vista_install( rtrt )
    #endif( rtrt_COMMON_BUILD )

vista_create_cmake_configs( rtrt )
vista_create_default_info_file( rtrt )
endif(USE_VISTA)

