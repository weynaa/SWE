#include "codelets.cuh"
#include "SWE_StarPU_Block.h"
#include <starpu/SWE_HUV_Matrix.h>
#include <cuda_runtime.h>
#include "codelets.h"
#include <cfloat>
#include <iostream>

#if defined(SOLVER_AUGRIE)

#include "solvers/AugRieCUDA.h"

__device__
void waveSolverCuda(float_type hLeft, float_type hRight,float huLeft, float huRight,float bLeft, float bRight,float& hUpdateLeft, float & hUpdateRight, float & huUpdateLeft, float& huUpdateRight, float & maxWaveSpeed){
    float results[5];
    augRieComputeNetUpdates(hLeft,hRight,
            huLeft,huRight,
            bLeft,bRight,
            SWE_StarPU_Block::g,static_cast<real>(0.01), static_cast<real>(0.000001), static_cast<real>(0.0001), 10,results);
    hUpdateLeft = results[0];
    hUpdateRight = results[1];
    huUpdateLeft = results[2];
    huUpdateRight = results[3];
    maxWaveSpeed = results[4];
}

#endif

#define CUDA_THREADS_PER_BLOCK 64

__device__ static float atomicMin(float *address, float val) {
    int *address_as_i = (int *) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

template<BoundaryEdge side>
__global__
void
computeNumericalFluxes_border(SWE_HUV_Matrix_interface mainBlock, SWE_HUV_Matrix_interface neighbourBlock,
                              starpu_matrix_interface b, float *maxTimestep, SWE_HUV_Matrix_interface netUpdates,
                              const uint32_t n,
                              float dX_inv, float dY_inv, float dX, float dY) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float maxEdgeSpeed[CUDA_THREADS_PER_BLOCK];
    if (index >= n) {
        return;
    }
    float_type hNetUpNeighbour, hNetUpMain;
    float_type huNetUpNeighbour, huNetUpMain;

    const uint32_t mainBlockX = side == BND_LEFT ? 0 : (side == BND_RIGHT ? mainBlock.nX - 1 : index);
    const uint32_t mainBlockY = side == BND_TOP ? 0 : (side == BND_BOTTOM ? mainBlock.nY - 1 : index);
    const uint32_t neighbourX = side == BND_LEFT || side == BND_RIGHT ? 0 : index;
    const uint32_t neighbourY = side == BND_TOP || BND_BOTTOM ? 0 : index;
    float_type hNeighbour = STARPU_SWE_HUV_MATRIX_GET_H_VAL(&neighbourBlock, neighbourX, neighbourY);
    float_type hMain = STARPU_SWE_HUV_MATRIX_GET_H_VAL(&mainBlock, mainBlockY, mainBlockX);
    float_type huNeighbour = STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&neighbourBlock, neighbourX, neighbourY);
    float_type huMain = STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&mainBlock, mainBlockY, mainBlockX);
    float_type bNeighbour = ((float_type * )
            STARPU_MATRIX_GET_PTR(&b))[
            (mainBlockY + (side == BND_BOTTOM ? 2 : (side == BND_TOP ? 0 : 1))) * STARPU_MATRIX_GET_LD(&b) +
            (mainBlockX + (side == BND_RIGHT ? 2 : (side == BND_LEFT ? 0 : 1)))];
    float_type bMain = ((float_type * )
            STARPU_MATRIX_GET_PTR(&b))[(mainBlockY + 1) * STARPU_MATRIX_GET_LD(&b) + (mainBlockX + 1)];


#if defined(SOLVER_AUGRIE)
    waveSolverCuda(
                        hNeighbour,  hMain,
                        huNeighbour, huMain,
                        bNeighbour,  bMain,
                        hNetUpNeighbour, hNetUpMain,
                        huNetUpNeighbour, huNetUpMain,
                        maxEdgeSpeed[threadIdx.x]
                       );
#endif


    if (side == BND_RIGHT || side == BND_LEFT) {
        STARPU_SWE_HUV_MATRIX_GET_H_VAL(&netUpdates, mainBlockY, mainBlockX) += dX_inv * hNetUpMain;
        STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&netUpdates, mainBlockY, mainBlockX) += dX_inv * huNetUpMain;
    } else {
        STARPU_SWE_HUV_MATRIX_GET_H_VAL(&netUpdates, mainBlockY, mainBlockX) += dY_inv * hNetUpMain;
        STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&netUpdates, mainBlockY, mainBlockX) += dY_inv * huNetUpMain;
    }
    __syncthreads();
    //Block wide reduction using shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            maxEdgeSpeed[threadIdx.x] = fmax(maxEdgeSpeed[threadIdx.x], maxEdgeSpeed[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float localMaxTimeStep = FLT_MAX;
        if (maxEdgeSpeed[0] > 0) {
            localMaxTimeStep = fmin(dX / maxEdgeSpeed[0], dY / maxEdgeSpeed[0]) * SWECodelets::CFL_NUMBER;
        }
        atomicMin(maxTimestep, localMaxTimeStep);
    }

}

__global__
void computeNumericalFluxes_mainBlock(SWE_HUV_Matrix_interface mainBlock, starpu_matrix_interface b, float *maxTimestep,
                                      SWE_HUV_Matrix_interface netUpdates,
                                      float dX_inv, float dY_inv, float dX, float dY) {
    const auto idxX = blockIdx.x * blockDim.x + threadIdx.x;
    const auto idxY = blockIdx.y * blockDim.y + threadIdx.y;

    const auto threadIdxLin = threadIdx.y*blockDim.x+threadIdx.x;

    __shared__ float maxEdgeSpeed[CUDA_THREADS_PER_BLOCK];

    if (idxX >= mainBlock.nX-1 || idxY >= mainBlock.nY-1) {
        return;
    }
    float_type hNetUpLeft, hNetUpRight;
    float_type huNetUpLeft, huNetUpRight;

    float_type hLeft = STARPU_SWE_HUV_MATRIX_GET_H_VAL(&mainBlock, idxX, idxY);
    float_type hRight = STARPU_SWE_HUV_MATRIX_GET_H_VAL(&mainBlock, idxX+1, idxY);
    float_type huLeft = STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&mainBlock, idxX, idxY);
    float_type huRight = STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&mainBlock, idxX+1, idxY);

    float_type bLeft = ((float_type*)(b.ptr))[(idxY+1)*b.ld+idxX+1];
    float_type bRight = ((float_type*)(b.ptr))[(idxY+1)*b.ld+idxX+2];

#if defined(SOLVER_AUGRIE)
    waveSolverCuda(
                        hLeft,  hRight,
                        huLeft, huRight,
                        bLeft, bRight,
                        hNetUpLeft, hNetUpRight,
                        huNetUpLeft, huNetUpRight,
                        maxEdgeSpeed[threadIdxLin]
                       );
#endif

    STARPU_SWE_HUV_MATRIX_GET_H_VAL(&netUpdates, idxX, idxY) += dX_inv*hNetUpLeft;
    STARPU_SWE_HUV_MATRIX_GET_H_VAL(&netUpdates, idxX+1, idxY) += dX_inv*hNetUpRight;
    STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&netUpdates, idxX, idxY) += dX_inv*huNetUpLeft;
    STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&netUpdates, idxX+1, idxY) += dX_inv*huNetUpRight;

    float_type hUpper = STARPU_SWE_HUV_MATRIX_GET_H_VAL(&mainBlock, idxX, idxY);
    float_type hLower = STARPU_SWE_HUV_MATRIX_GET_H_VAL(&mainBlock, idxX, idxY+1);
    float_type hvUpper = STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&mainBlock, idxX, idxY);
    float_type hvLower = STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&mainBlock, idxX, idxY+1);

    float_type bUpper = ((float_type*)(b.ptr))[(idxY+1)*b.ld+idxX+1];
    float_type bLower = ((float_type*)(b.ptr))[(idxY+2)*b.ld+idxX+1];
    float l_maxEdgeSpeed;
#if defined(SOLVER_AUGRIE)
    waveSolverCuda(
                        hUpper,  hLower,
                        hvUpper, hvLower,
                        bUpper, bLower,
                        hNetUpLeft, hNetUpRight,
                        huNetUpLeft, huNetUpRight,
                        l_maxEdgeSpeed
                       );
#endif
    maxEdgeSpeed[threadIdxLin] = fmax( maxEdgeSpeed[threadIdxLin],l_maxEdgeSpeed);

    STARPU_SWE_HUV_MATRIX_GET_H_VAL(&mainBlock, idxX, idxY) += dY_inv*hNetUpLeft;
    STARPU_SWE_HUV_MATRIX_GET_H_VAL(&mainBlock, idxX, idxY+1) += dY_inv*hNetUpRight;
    STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&mainBlock, idxX, idxY)+= dY_inv*huNetUpLeft;
    STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&mainBlock, idxX, idxY+1)+= dY_inv*huNetUpRight;

    __syncthreads();
    //Block wide reduction using shared memory
    for (unsigned int s = (blockDim.x*blockDim.y) / 2; s > 0; s >>= 1) {
        if (threadIdxLin < s) {
            maxEdgeSpeed[threadIdxLin] = fmax(maxEdgeSpeed[threadIdxLin], maxEdgeSpeed[threadIdxLin + s]);
        }
        __syncthreads();
    }
    if (threadIdxLin == 0) {
        float localMaxTimeStep = FLT_MAX;
        if (maxEdgeSpeed[0] > 0) {
            localMaxTimeStep = fmin(dX / maxEdgeSpeed[0], dY / maxEdgeSpeed[0]) * SWECodelets::CFL_NUMBER;
        }
        atomicMin(maxTimestep, localMaxTimeStep);
    }

}

void computeNumericalFluxes_cuda(void *buffers[], void *cl_arg) {
    const SWE_StarPU_Block *pBlock;
    starpu_codelet_unpack_args(cl_arg, &pBlock);

    const auto mainBlock = buffers[0];
    const auto leftGhost = buffers[1 + BND_LEFT];
    const auto rightGhost = buffers[1 + BND_RIGHT];
    const auto bottomGhost = buffers[1 + BND_BOTTOM];
    const auto topGhost = buffers[1 + BND_TOP];
    const auto b = buffers[5];
    const auto netUpdates = buffers[6];

    float *maxTimestep = (float *) STARPU_VARIABLE_GET_PTR(buffers[7]);

    const auto nX = pBlock->getNx();
    const auto nY = pBlock->getNy();

    const auto dX = pBlock->getDx();
    const auto dY = pBlock->getDy();

    const auto dX_inv = 1 / dX;
    const auto dY_inv = 1 / dY;

    const cudaStream_t stream = starpu_cuda_get_local_stream();

    cudaMemsetAsync(STARPU_SWE_HUV_MATRIX_GET_H_PTR(netUpdates), 0,
                    sizeof(float_type) * nX * nY, stream);
    cudaMemsetAsync(STARPU_SWE_HUV_MATRIX_GET_HU_PTR(netUpdates), 0,
                    sizeof(float_type) * nX * nY, stream);
    cudaMemsetAsync(STARPU_SWE_HUV_MATRIX_GET_HV_PTR(netUpdates), 0,
                    sizeof(float_type) * nX * nY, stream);

    uint32_t gridSize = (nX + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
    computeNumericalFluxes_border<BND_LEFT>
    <<<gridSize, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(mainBlock),
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(leftGhost),
            *((starpu_matrix_interface *) (b)),
            maxTimestep,
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(netUpdates),
            nY,
            dX_inv,
            dY_inv,
            dX,
            dY
    );
    computeNumericalFluxes_border<BND_RIGHT>
    <<<gridSize, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(mainBlock),
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(rightGhost),
            *((starpu_matrix_interface *) (b)),
            maxTimestep,
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(netUpdates),
            nY,
            dX_inv,
            dY_inv,
            dX,
            dY
    );

    computeNumericalFluxes_border<BND_TOP>
    <<<gridSize, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(mainBlock),
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(topGhost),
            *((starpu_matrix_interface *) (b)),
            maxTimestep,
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(netUpdates),
            nX,
            dX_inv,
            dY_inv,
            dX,
            dY
    );
    computeNumericalFluxes_border<BND_BOTTOM>
    <<<gridSize, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(mainBlock),
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(bottomGhost),
            *((starpu_matrix_interface *) (b)),
            maxTimestep,
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(netUpdates),
            nX,
            dX_inv,
            dY_inv,
            dX,
            dY
    );

    const auto blockWidth = std::floor(std::sqrt(CUDA_THREADS_PER_BLOCK));
    dim3 threads(blockWidth,blockWidth);
    dim3 blocks(
            (nX+threads.x-1)/threads.x,
            (nY+threads.y-1)/threads.y);
    computeNumericalFluxes_mainBlock<<<blocks,threads,0,stream>>>(
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(mainBlock),
            *((starpu_matrix_interface *) (b)),
            maxTimestep,
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(netUpdates),
            dX_inv,
            dY_inv,
            dX,
            dY
            );

    cudaStreamSynchronize(stream);

}

__global__
void variableMin_cuda_kernel(float *a, float *b) {
    *a = std::min(*a, *b);
}

void variableMin_cuda(void *buffers[], void *cl_args) {
    float *a = (float *) STARPU_VARIABLE_GET_PTR(buffers[0]);
    float *b = (float *) STARPU_VARIABLE_GET_PTR(buffers[1]);
    const auto stream = starpu_cuda_get_local_stream();
    variableMin_cuda_kernel<<<1, 1, 0, stream>>>(a, b);
    cudaStreamSynchronize(stream);
}

__global__
void variableSetInf_cuda_kernel(float *value) {
    *value = INFINITY;
}

void variableSetInf_cuda(void *buffers[], void *cl_args) {
    float *value = (float *) STARPU_VARIABLE_GET_PTR(buffers[0]);
    const auto stream = starpu_cuda_get_local_stream();
    variableSetInf_cuda_kernel<<<1, 1, 0, stream>>>(value);
    cudaStreamSynchronize(stream);
}

