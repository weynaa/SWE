#include "codelets.cuh"
#include "SWE_StarPU_Block.h"
#include <starpu/SWE_HUV_Matrix.h>
#include <cuda_runtime.h>
#include "codelets.h"
#include "../../starpu/SWE_HUV_Matrix.h"
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
    maxEdgeSpeed[threadIdx.x] = 0;
    if (index < n) {
        if (side == BND_RIGHT || side == BND_LEFT) {
            float_type hNetUpLeft, hNetUpRight;
            float_type huNetUpLeft, huNetUpRight;

            float_type hLeft = STARPU_SWE_HUV_MATRIX_GET_H_VAL(
                    side == BND_LEFT ? &neighbourBlock : &mainBlock,
                    side == BND_LEFT ? 0 : (mainBlock.nX - 1),
                    index);
            float_type hRight = STARPU_SWE_HUV_MATRIX_GET_H_VAL(
                    side == BND_LEFT ? &mainBlock : &neighbourBlock,
                    0,
                    index);
            float_type huLeft = STARPU_SWE_HUV_MATRIX_GET_HU_VAL(
                    side == BND_LEFT ? &neighbourBlock : &mainBlock,
                    side == BND_LEFT ? 0 : (mainBlock.nX - 1),
                    index);
            float_type huRight = STARPU_SWE_HUV_MATRIX_GET_HU_VAL(
                    side == BND_LEFT ? &mainBlock : &neighbourBlock,
                    0,
                    index);
            float_type bLeft = ((float_type *)
                    STARPU_MATRIX_GET_PTR(&b))[
                    (index + 1) * STARPU_MATRIX_GET_LD(&b) +
                    (side == BND_LEFT ? 0 : mainBlock.nX)];
            float_type bRight = ((float_type *)
                    STARPU_MATRIX_GET_PTR(&b))[(index + 1) * STARPU_MATRIX_GET_LD(&b) + 1 +
                                               (side == BND_LEFT ? 0 : mainBlock.nX)];


            waveSolverCuda(
                    hLeft, hRight,
                    huLeft, huRight,
                    bLeft, bRight,
                    hNetUpLeft, hNetUpRight,
                    huNetUpLeft, huNetUpRight,
                    maxEdgeSpeed[threadIdx.x]
            );

            if (side == BND_LEFT) {
                STARPU_SWE_HUV_MATRIX_GET_H_VAL(&netUpdates, 0, index) += dX_inv * hNetUpRight;
                STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&netUpdates, 0, index) += dX_inv * huNetUpRight;
            } else {
                STARPU_SWE_HUV_MATRIX_GET_H_VAL(&netUpdates, netUpdates.nX - 1, index) += dX_inv * hNetUpLeft;
                STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&netUpdates, netUpdates.nX - 1, index) += dX_inv * huNetUpLeft;
            }
        } else {
            float_type hNetUpUpper, hNetUpLower;
            float_type hvNetUpUpper, hvNetUpLower;

            float_type hUpper = STARPU_SWE_HUV_MATRIX_GET_H_VAL(
                    side == BND_TOP ? &neighbourBlock : &mainBlock,
                    index + (side == BND_TOP ? 1 : 0),
                    side == BND_TOP ? 0 : (mainBlock.nY - 1)
            );
            float_type hLower = STARPU_SWE_HUV_MATRIX_GET_H_VAL(
                    side == BND_TOP ? &mainBlock : &neighbourBlock,
                    side == BND_TOP ? index : (index + 1),
                    0
            );
            float_type hvUpper = STARPU_SWE_HUV_MATRIX_GET_HV_VAL(
                    side == BND_TOP ? &neighbourBlock : &mainBlock,
                    index + (side == BND_TOP ? 1 : 0),
                    side == BND_TOP ? 0 : (mainBlock.nY - 1)
            );
            float_type hvLower = STARPU_SWE_HUV_MATRIX_GET_HV_VAL(
                    side == BND_TOP ? &mainBlock : &neighbourBlock,
                    side == BND_TOP ? index : (index + 1),
                    0
            );

            float_type bUpper = ((float_type *)
                    STARPU_MATRIX_GET_PTR(&b))[(side == BND_TOP ? 0 : mainBlock.nY) *
                                               STARPU_MATRIX_GET_LD(&b) + (index + 1)];
            float_type bLower = ((float_type *)
                    STARPU_MATRIX_GET_PTR(&b))[((side == BND_TOP ? 0 : mainBlock.nY) + 1) *
                                               STARPU_MATRIX_GET_LD(&b) + (index + 1)];
            waveSolverCuda(hUpper, hLower,
                           hvUpper, hvLower,
                           bUpper, bLower,
                           hNetUpUpper, hNetUpLower,
                           hvNetUpUpper, hvNetUpLower,
                           maxEdgeSpeed[threadIdx.x]);
            if (side == BND_TOP) {
                STARPU_SWE_HUV_MATRIX_GET_H_VAL(&netUpdates, index, 0) += dY_inv * hNetUpLower;
                STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&netUpdates, index, 0) += dY_inv * hvNetUpLower;
            } else {
                STARPU_SWE_HUV_MATRIX_GET_H_VAL(&netUpdates, index, mainBlock.nY - 1) += dY_inv * hNetUpUpper;
                STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&netUpdates, index, mainBlock.nY - 1) += dY_inv * hvNetUpUpper;
            }
        }
    }

    __syncthreads();
//Block wide reduction using shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && threadIdx.x + s < CUDA_THREADS_PER_BLOCK) {
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

    const auto threadIdxLin = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ float maxEdgeSpeed[CUDA_THREADS_PER_BLOCK];
    maxEdgeSpeed[threadIdxLin] = 0;
    if (idxX > 0 && idxX < mainBlock.nX && idxY < mainBlock.nY) {
        float_type hNetUpLeft, hNetUpRight;
        float_type huNetUpLeft, huNetUpRight;

        float_type hLeft = STARPU_SWE_HUV_MATRIX_GET_H_VAL(&mainBlock, idxX - 1, idxY);
        float_type hRight = STARPU_SWE_HUV_MATRIX_GET_H_VAL(&mainBlock, idxX, idxY);
        float_type huLeft = STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&mainBlock, idxX - 1, idxY);
        float_type huRight = STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&mainBlock, idxX, idxY);

        float_type bLeft = ((float_type *) (b.ptr))[(idxY + 1) * b.ld + idxX];
        float_type bRight = ((float_type *) (b.ptr))[(idxY + 1) * b.ld + idxX + 1];
        float l_maxEdgeSpeed;
        waveSolverCuda(
                hLeft, hRight,
                huLeft, huRight,
                bLeft, bRight,
                hNetUpLeft, hNetUpRight,
                huNetUpLeft, huNetUpRight,
                l_maxEdgeSpeed
        );
        maxEdgeSpeed[threadIdxLin] = fmax(maxEdgeSpeed[threadIdxLin], l_maxEdgeSpeed);

        STARPU_SWE_HUV_MATRIX_GET_H_VAL(&netUpdates, idxX - 1, idxY) += dX_inv * hNetUpLeft;
        STARPU_SWE_HUV_MATRIX_GET_H_VAL(&netUpdates, idxX, idxY) += dX_inv * hNetUpRight;
        STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&netUpdates, idxX - 1, idxY) += dX_inv * huNetUpLeft;
        STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&netUpdates, idxX, idxY) += dX_inv * huNetUpRight;
    }
    if (idxX < mainBlock.nX && idxY < mainBlock.nY - 1) {
        float_type hUpper = STARPU_SWE_HUV_MATRIX_GET_H_VAL(&mainBlock, idxX, idxY);
        float_type hLower = STARPU_SWE_HUV_MATRIX_GET_H_VAL(&mainBlock, idxX, idxY + 1);
        float_type hvUpper = STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&mainBlock, idxX, idxY);
        float_type hvLower = STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&mainBlock, idxX, idxY + 1);

        float_type bUpper = ((float_type *) (b.ptr))[(idxY + 1) * b.ld + idxX + 1];
        float_type bLower = ((float_type *) (b.ptr))[(idxY + 2) * b.ld + idxX + 1];

        float_type hNetUpUpper, hNetUpLower;
        float_type hvNetUpUpper, hvNetUpLower;
        float l_maxEdgeSpeed;
        waveSolverCuda(
                hUpper, hLower,
                hvUpper, hvLower,
                bUpper, bLower,
                hNetUpUpper, hNetUpLower,
                hvNetUpUpper, hvNetUpLower,
                l_maxEdgeSpeed
        );

        maxEdgeSpeed[threadIdxLin] = fmax(maxEdgeSpeed[threadIdxLin], l_maxEdgeSpeed);

        STARPU_SWE_HUV_MATRIX_GET_H_VAL(&netUpdates, idxX, idxY) += dY_inv * hNetUpUpper;
        STARPU_SWE_HUV_MATRIX_GET_H_VAL(&netUpdates, idxX, idxY + 1) += dY_inv * hNetUpLower;
        STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&netUpdates, idxX, idxY) += dY_inv * hvNetUpUpper;
        STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&netUpdates, idxX, idxY + 1) += dY_inv * hvNetUpLower;
    }
    __syncthreads();
    //Block wide reduction using shared memory
    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
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
            *((starpu_matrix_interface * )(b)),
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
            *((starpu_matrix_interface * )(b)),
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
            *((starpu_matrix_interface * )(b)),
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
            *((starpu_matrix_interface * )(b)),
            maxTimestep,
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(netUpdates),
            nX,
            dX_inv,
            dY_inv,
            dX,
            dY
    );

    const auto blockWidth = std::floor(std::sqrt(CUDA_THREADS_PER_BLOCK));
    dim3 threads(blockWidth, blockWidth);
    dim3 blocks(
            (nX + threads.x - 1) / threads.x,
            (nY + threads.y - 1) / threads.y);
    computeNumericalFluxes_mainBlock<<<blocks, threads, 0, stream>>>(
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(mainBlock),
            *((starpu_matrix_interface * )(b)),
            maxTimestep,
            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(netUpdates),
            dX_inv,
            dY_inv,
            dX,
            dY
    );
}

__global__
void variableMin_cuda_kernel(float *a, float *b) {
    *a = fmin(*a, *b);
}

void variableMin_cuda(void *buffers[], void *cl_args) {
    float *a = (float *) STARPU_VARIABLE_GET_PTR(buffers[0]);
    float *b = (float *) STARPU_VARIABLE_GET_PTR(buffers[1]);
    const auto stream = starpu_cuda_get_local_stream();
    variableMin_cuda_kernel<<<1, 1, 0, stream>>>(a, b);
}

__global__
void variableSetInf_cuda_kernel(float *value) {
    *value = INFINITY;
}

void variableSetInf_cuda(void *buffers[], void *cl_args) {
    float *value = (float *) STARPU_VARIABLE_GET_PTR(buffers[0]);
    const auto stream = starpu_cuda_get_local_stream();
    variableSetInf_cuda_kernel<<<1, 1, 0, stream>>>(value);
}

__global__
void updateUnkowns_cuda_kernel(const SWE_HUV_Matrix_interface myBlock, const SWE_HUV_Matrix_interface updates,
                               const float *const dt,
                               const size_t nX,
                               const size_t nY) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nX || y >= nY) {
        return;
    }
    STARPU_SWE_HUV_MATRIX_GET_H_VAL(&myBlock, x, y) -= *dt * STARPU_SWE_HUV_MATRIX_GET_H_VAL(&updates, x, y);
    STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&myBlock, x, y) -= *dt * STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&updates, x, y);
    STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&myBlock, x, y) -= *dt * STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&updates, x, y);
    if (STARPU_SWE_HUV_MATRIX_GET_H_VAL(&myBlock, x, y) < SWECodelets::DRY_LIMIT) {
        STARPU_SWE_HUV_MATRIX_GET_H_VAL(&myBlock, x, y) =
        STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&myBlock, x, y) =
        STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&myBlock, x, y) = 0;
    }
}

void updateUnknowns_cuda(void *buffers[], void *cl_args) {
    const SWE_StarPU_Block *pBlock;
    starpu_codelet_unpack_args(cl_args, &pBlock);

    const auto myBlock = buffers[0];
    const auto updates = buffers[1];
    const auto dt = (const float *) STARPU_VARIABLE_GET_PTR(buffers[2]);

    const cudaStream_t stream = starpu_cuda_get_local_stream();

    const auto blockWidth = std::floor(std::sqrt(CUDA_THREADS_PER_BLOCK));
    dim3 threads(blockWidth, blockWidth);
    dim3 blocks(
            (pBlock->getNx() + threads.x - 1) / threads.x,
            (pBlock->getNy() + threads.y - 1) / threads.y);

    updateUnkowns_cuda_kernel<<<blocks, threads, 0, stream>>>(STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlock),
                                                              STARPU_SWE_HUV_MATRIX_GET_INTERFACE(updates),
                                                              dt,
                                                              pBlock->getNx(),
                                                              pBlock->getNy());
}


template<BoundaryEdge side, BoundaryType boundaryType>
__global__
void updateGhostLayers_cuda_kernel(SWE_HUV_Matrix_interface myBlockData, SWE_HUV_Matrix_interface myBorderData,
                                   SWE_HUV_Matrix_interface myNeighbourData) {
    constexpr bool vertical = side == BND_LEFT || side == BND_RIGHT;
    const auto nx = STARPU_SWE_HUV_MATRIX_GET_NX(&myBlockData);
    const auto ny = STARPU_SWE_HUV_MATRIX_GET_NY(&myBlockData);

    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (vertical ? ny : nx)) {
        return;
    }
    switch (boundaryType) {
        case WALL:
        case OUTFLOW: {
            const size_t outerX = vertical ? 0 : i + 1;
            const size_t innerX = side == BND_LEFT ? 0 : (side == BND_RIGHT ? nx - 1 : i);
            const size_t outerY = vertical ? i : 0;
            const size_t innerY = side == BND_TOP ? 0 : (side == BND_BOTTOM ? ny - 1 : i);

            STARPU_SWE_HUV_MATRIX_GET_H_VAL(&myBorderData, outerX, outerY) =
                    STARPU_SWE_HUV_MATRIX_GET_H_VAL(&myBlockData, innerX, innerY);
            STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&myBorderData, outerX, outerY) =
                    (vertical && boundaryType == WALL ? -1.f : 1.f) *
                    STARPU_SWE_HUV_MATRIX_GET_HU_VAL(
                            &myBlockData, innerX, innerY);
            STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&myBorderData, outerX, outerY) =
                    (!vertical && boundaryType == WALL ? -1.f : 1.f) *
                    STARPU_SWE_HUV_MATRIX_GET_HV_VAL(
                            &myBlockData, innerX, innerY);
        }
            break;
        case CONNECT: {
            const auto neighbourNX = STARPU_SWE_HUV_MATRIX_GET_NX(&myNeighbourData);
            const auto neighbourNY = STARPU_SWE_HUV_MATRIX_GET_NY(&myNeighbourData);
            const auto boundaryX = vertical ? 0 : 1 + i;
            const auto boundaryY = vertical ? i : 0;

            const auto neighbourX = side == BND_LEFT ? neighbourNX - 1 : (side == BND_RIGHT ? 0 : i);
            const auto neigbhourY = side == BND_TOP ? neighbourNY - 1 : (side == BND_BOTTOM ? 0 : i);

            STARPU_SWE_HUV_MATRIX_GET_H_VAL(&myBorderData, boundaryX, boundaryY) =
                    STARPU_SWE_HUV_MATRIX_GET_H_VAL(&myNeighbourData, neighbourX, neigbhourY);
            STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&myBorderData, boundaryX, boundaryY) =
                    STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&myNeighbourData, neighbourX, neigbhourY);
            STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&myBorderData, boundaryX, boundaryY) =
                    STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&myNeighbourData, neighbourX, neigbhourY);
        }
            break;
        case PASSIVE:
            break;
        default:
            //printf("unkown boundary type");
            break;
    }

    if (i == 0) {
        if (side == BND_TOP) {
            STARPU_SWE_HUV_MATRIX_GET_H_VAL(&myBorderData, 0, 0) =
                    STARPU_SWE_HUV_MATRIX_GET_H_VAL(&myBlockData, 0, 0);
            STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&myBorderData, 0, 0) =
                    STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&myBlockData, 0, 0);
            STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&myBorderData, 0, 0) =
                    STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&myBlockData, 0, 0);
        } else if (side == BND_BOTTOM) {
            STARPU_SWE_HUV_MATRIX_GET_H_VAL(&myBorderData, 0, 0) =
                    STARPU_SWE_HUV_MATRIX_GET_H_VAL(&myBlockData, 0, ny - 1);
            STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&myBorderData, 0, 0) =
                    STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&myBlockData, 0, ny - 1);
            STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&myBorderData, 0, 0) =
                    STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&myBlockData, 0, ny - 1);
        }
    } else if (i == (vertical ? ny : nx) - 1) {
        if (side == BND_TOP) {
            STARPU_SWE_HUV_MATRIX_GET_H_VAL(&myBorderData, nx + 1, 0) =
                    STARPU_SWE_HUV_MATRIX_GET_H_VAL(&myBlockData, nx - 1, 0);
            STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&myBorderData, nx + 1, 0) =
                    STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&myBlockData, nx - 1, 0);
            STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&myBorderData, nx + 1, 0) =
                    STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&myBlockData, nx - 1, 0);
        } else if (side == BND_BOTTOM) {
            STARPU_SWE_HUV_MATRIX_GET_H_VAL(&myBorderData, nx + 1, 0) =
                    STARPU_SWE_HUV_MATRIX_GET_H_VAL(&myBlockData, nx - 1, ny - 1);
            STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&myBorderData, nx + 1, 0) =
                    STARPU_SWE_HUV_MATRIX_GET_HU_VAL(&myBlockData, nx - 1, ny - 1);
            STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&myBorderData, nx + 1, 0) =
                    STARPU_SWE_HUV_MATRIX_GET_HV_VAL(&myBlockData, nx - 1, ny - 1);
        }
    }

}

void updateGhostLayers_cuda(void *buffers[], void *cl_arg) {
    const SWE_StarPU_Block *thisBlock;
    BoundaryEdge side;
    starpu_codelet_unpack_args(cl_arg, &side, &thisBlock);
#ifdef DBG
    cout << "Set simple boundary conditions " << endl << flush;
#endif
    auto myBlockData = buffers[1];
    auto myBorderData = buffers[0];
    auto myNeighbourData = buffers[2];

    const auto nx = STARPU_SWE_HUV_MATRIX_GET_NX(myBlockData);
    const auto ny = STARPU_SWE_HUV_MATRIX_GET_NY(myBlockData);
    const bool vertical = side == BND_LEFT || side == BND_RIGHT;

    const cudaStream_t stream = starpu_cuda_get_local_stream();

    dim3 threads(CUDA_THREADS_PER_BLOCK);
    dim3 blocks(((vertical ? ny : nx) + threads.x - 1) / threads.x);

    switch (side) {
        case BND_LEFT:
            switch (thisBlock->boundary[BND_LEFT]) {
                case PASSIVE:
                    updateGhostLayers_cuda_kernel<BND_LEFT, PASSIVE><<<blocks, threads, 0, stream>>>(
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlockData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBorderData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myNeighbourData)
                    );
                    break;
                case WALL:
                    updateGhostLayers_cuda_kernel<BND_LEFT, WALL><<<blocks, threads, 0, stream>>>(
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlockData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBorderData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myNeighbourData)
                    );
                    break;
                case CONNECT:
                    updateGhostLayers_cuda_kernel<BND_LEFT, CONNECT><<<blocks, threads, 0, stream>>>(
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlockData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBorderData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myNeighbourData)
                    );
                    break;
                case OUTFLOW:
                    updateGhostLayers_cuda_kernel<BND_LEFT, OUTFLOW><<<blocks, threads, 0, stream>>>(
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlockData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBorderData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myNeighbourData)
                    );
                    break;
            }
            break;
        case BND_RIGHT:
            switch (thisBlock->boundary[BND_RIGHT]) {
                case PASSIVE:
                    updateGhostLayers_cuda_kernel<BND_RIGHT, PASSIVE><<<blocks, threads, 0, stream>>>(
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlockData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBorderData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myNeighbourData)
                    );
                    break;
                case WALL:
                    updateGhostLayers_cuda_kernel<BND_RIGHT, WALL><<<blocks, threads, 0, stream>>>(
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlockData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBorderData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myNeighbourData)
                    );
                    break;
                case CONNECT:
                    updateGhostLayers_cuda_kernel<BND_RIGHT, CONNECT><<<blocks, threads, 0, stream>>>(
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlockData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBorderData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myNeighbourData)
                    );
                    break;
                case OUTFLOW:
                    updateGhostLayers_cuda_kernel<BND_RIGHT, OUTFLOW><<<blocks, threads, 0, stream>>>(
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlockData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBorderData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myNeighbourData)
                    );
                    break;
            }
            break;
        case BND_TOP:
            switch (thisBlock->boundary[BND_TOP]) {
                case PASSIVE:
                    updateGhostLayers_cuda_kernel<BND_TOP, PASSIVE><<<blocks, threads, 0, stream>>>(
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlockData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBorderData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myNeighbourData)
                    );
                    break;
                case WALL:
                    updateGhostLayers_cuda_kernel<BND_TOP, WALL><<<blocks, threads, 0, stream>>>(
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlockData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBorderData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myNeighbourData)
                    );
                    break;
                case CONNECT:
                    updateGhostLayers_cuda_kernel<BND_TOP, CONNECT><<<blocks, threads, 0, stream>>>(
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlockData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBorderData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myNeighbourData)
                    );
                    break;
                case OUTFLOW:
                    updateGhostLayers_cuda_kernel<BND_TOP, OUTFLOW><<<blocks, threads, 0, stream>>>(
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlockData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBorderData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myNeighbourData)
                    );
                    break;
            }
            break;
        case BND_BOTTOM:
            switch (thisBlock->boundary[BND_BOTTOM]) {
                case PASSIVE:
                    updateGhostLayers_cuda_kernel<BND_BOTTOM, PASSIVE><<<blocks, threads, 0, stream>>>(
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlockData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBorderData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myNeighbourData)
                    );
                    break;
                case WALL:
                    updateGhostLayers_cuda_kernel<BND_BOTTOM, WALL><<<blocks, threads, 0, stream>>>(
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlockData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBorderData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myNeighbourData)
                    );
                    break;
                case CONNECT:
                    updateGhostLayers_cuda_kernel<BND_BOTTOM, CONNECT><<<blocks, threads, 0, stream>>>(
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlockData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBorderData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myNeighbourData)
                    );
                    break;
                case OUTFLOW:
                    updateGhostLayers_cuda_kernel<BND_BOTTOM, OUTFLOW><<<blocks, threads, 0, stream>>>(
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBlockData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myBorderData),
                            STARPU_SWE_HUV_MATRIX_GET_INTERFACE(myNeighbourData)
                    );
                    break;
            }
            break;
    }
}
