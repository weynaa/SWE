#include "testkernels.cuh"
#include <cstdlib>
#define STARPU_USE_CUDA
#include <starpu.h>

__global__ void  test_cuda_kernel(float *val, float factor, size_t nx, size_t ny, size_t stride) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < nx && y < ny) {
        val[y*stride+x] *= factor;
    }
}

void test_cuda_func(void *buffers[], void *args) {
    float *factor = (float *) args;
    /* length of the vector */
    unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
    unsigned ny = STARPU_MATRIX_GET_NY(buffers[0]);
    unsigned row_stride = STARPU_MATRIX_GET_LD(buffers[0]);
    /* local copy of the vector pointer */
    float *val = (float *) STARPU_MATRIX_GET_PTR(buffers[0]);
    dim3 threads_per_block = {8,8};
    dim3 nblocks = {(nx + threads_per_block.x - 1) / threads_per_block.x,
                    (ny + threads_per_block.y - 1) / threads_per_block.y};
    test_cuda_kernel<<<nblocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>>(val, *factor, nx,ny,row_stride);
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}