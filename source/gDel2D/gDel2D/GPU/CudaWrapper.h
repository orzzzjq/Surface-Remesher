#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

#if __CUDA_ARCH__ >= 200 && defined( CUDA_ERROR_CHECK )
#define CudaAssert(X)                                           \
    if ( !(X) )                                                 \
    {                                                           \
        printf( "!!!Thread %d:%d failed assert at %s:%d!!!\n",  \
            blockIdx.x, threadIdx.x, __FILE__, __LINE__ );      \
    } 
#else
#define CudaAssert(X) 
#endif

template< typename T >
T* cuNew( int num )
{
    T* loc              = NULL;
    const size_t space  = num * sizeof( T );
    CudaSafeCall( cudaMalloc( &loc, space ) );

    return loc;
}

template< typename T >
void cuDelete( T** loc )
{
    CudaSafeCall( cudaFree( *loc ) );
    *loc = NULL;
    return;
}

template< typename T >
__forceinline__ __device__ void cuSwap( T& v0, T& v1 )
{
    const T tmp = v0;
    v0          = v1;
    v1          = tmp;

    return;
}

inline void cuPrintMemory( const char* inStr )
{
    const int MegaByte = ( 1 << 20 );

    size_t free;
    size_t total;
    CudaSafeCall( cudaMemGetInfo( &free, &total ) );

    printf( "[%s] Memory used: %d MB\n", inStr, ( total - free ) / MegaByte );

    return;
}

// Obtained from: C:\ProgramData\NVIDIA Corporation\GPU SDK\C\common\inc\cutil_inline_runtime.h
// This function returns the best GPU (with maximum GFLOPS)
inline int cutGetMaxGflopsDeviceId()
{
    int current_device   = 0, sm_per_multiproc = 0;
    int max_compute_perf = 0, max_perf_device  = 0;
    int device_count     = 0, best_SM_arch     = 0;
    int arch_cores_sm[3] = { 1, 8, 32 };
    cudaDeviceProp deviceProp;

    cudaGetDeviceCount( &device_count );
    // Find the best major SM Architecture GPU device
    while ( current_device < device_count ) {
        cudaGetDeviceProperties( &deviceProp, current_device );
        if (deviceProp.major > 0 && deviceProp.major < 9999)
        {
            if ( deviceProp.major > best_SM_arch )
                best_SM_arch = deviceProp.major;
        }
        current_device++;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;
    while( current_device < device_count ) {
        cudaGetDeviceProperties( &deviceProp, current_device );
        if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
            sm_per_multiproc = 1;
        } else if (deviceProp.major <= 2) {
            sm_per_multiproc = arch_cores_sm[deviceProp.major];
        } else {
            sm_per_multiproc = arch_cores_sm[2];
        }

        int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
        if( compute_perf  > max_compute_perf ) {
            // If we find GPU with SM major > 2, search only these
            if ( best_SM_arch > 2 ) {
                // If our device==dest_SM_arch, choose this, or else pass
                if (deviceProp.major == best_SM_arch) { 
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                }
            } else {
                max_compute_perf  = compute_perf;
                max_perf_device   = current_device;
            }
        }
        ++current_device;
    }
    return max_perf_device;
}