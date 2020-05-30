
#include "codelets.h"
#include "SWE_StarPU_Block.h"
#include "writer/StarPUBlockWriter.h"
#include "SWE_StarPU_Sim.h"
#include <vector>
#include <algorithm>
#include <limits>

#ifdef ENABLE_CUDA
#include "codelets.cuh"
#endif


void updateGhostLayers_cpu(void *buffers[], void *cl_arg) {
    const SWE_StarPU_Block *thisBlock;
    BoundaryEdge side;
    starpu_codelet_unpack_args(cl_arg, &side, &thisBlock);
#ifdef DBG
    cout << "Set simple boundary conditions " << endl << flush;
#endif
    auto myBlockData = buffers[1];
    auto myBorderData = buffers[0];


    const bool vertical = side == BND_LEFT || side == BND_RIGHT;
    const auto nx = STARPU_SWE_HUV_MATRIX_GET_NX(myBlockData);
    const auto ny = STARPU_SWE_HUV_MATRIX_GET_NY(myBlockData);

    switch (thisBlock->boundary[side]) {
        case WALL:
        case OUTFLOW: {
            const bool wall = thisBlock->boundary[side] == WALL;
#ifdef VECTORIZE
#pragma omp simd
#endif
            for (size_t j = 0; j < (vertical ? ny : nx); j++) {
                const size_t outerX = vertical ? 0 : j + 1;
                const size_t innerX = side == BND_LEFT ? 0 : (side == BND_RIGHT ? nx - 1 : j);
                const size_t outerY = vertical ? j : 0;
                const size_t innerY = side == BND_TOP ? 0 : (side == BND_BOTTOM ? ny - 1 : j);

                STARPU_SWE_HUV_MATRIX_GET_H_VAL(myBorderData, outerX, outerY) =
                        STARPU_SWE_HUV_MATRIX_GET_H_VAL(myBlockData, innerX, innerY);
                STARPU_SWE_HUV_MATRIX_GET_HU_VAL(myBorderData, outerX, outerY) = (vertical && wall ? -1.f : 1.f) *
                                                                                 STARPU_SWE_HUV_MATRIX_GET_HU_VAL(
                                                                                         myBlockData, innerX, innerY);
                STARPU_SWE_HUV_MATRIX_GET_HV_VAL(myBorderData, outerX, outerY) = (!vertical && wall ? -1.f : 1.f) *
                                                                                 STARPU_SWE_HUV_MATRIX_GET_HV_VAL(
                                                                                         myBlockData, innerX, innerY);

            }
        }
            break;
        case CONNECT:
        case PASSIVE:
            break;
        default:
            assert(false);
            break;
    }

    if (thisBlock->boundary[side] == CONNECT) {
        auto myNeighbourData = buffers[2];
        const auto neighbourNX = STARPU_SWE_HUV_MATRIX_GET_NX(myNeighbourData);
        const auto neighbourNY = STARPU_SWE_HUV_MATRIX_GET_NY(myNeighbourData);
#ifdef VECTORIZE
#pragma omp simd
#endif
        for (size_t i = 0; i < (vertical ? ny : nx); ++i) {
            const size_t boundaryX = vertical ? 0 : 1 + i;
            const size_t boundaryY = vertical ? i : 0;

            const size_t neighbourX = side == BND_LEFT ? neighbourNX - 1 : (side == BND_RIGHT ? 0 : i);
            const size_t neigbhourY = side == BND_TOP ? neighbourNY - 1 : (side == BND_BOTTOM ? 0 : i);

            STARPU_SWE_HUV_MATRIX_GET_H_VAL(myBorderData, boundaryX, boundaryY) =
                    STARPU_SWE_HUV_MATRIX_GET_H_VAL(myNeighbourData, neighbourX, neigbhourY);
            STARPU_SWE_HUV_MATRIX_GET_HU_VAL(myBorderData, boundaryX, boundaryY) =
                    STARPU_SWE_HUV_MATRIX_GET_HU_VAL(myNeighbourData, neighbourX, neigbhourY);
            STARPU_SWE_HUV_MATRIX_GET_HV_VAL(myBorderData, boundaryX, boundaryY) =
                    STARPU_SWE_HUV_MATRIX_GET_HV_VAL(myNeighbourData, neighbourX, neigbhourY);
        }
    }
    //Update the corner values only the top and bottom boundary contain these
    if (side == BND_TOP) {
        STARPU_SWE_HUV_MATRIX_GET_H_VAL(myBorderData, 0, 0) =
                STARPU_SWE_HUV_MATRIX_GET_H_VAL(myBlockData, 0, 0);
        STARPU_SWE_HUV_MATRIX_GET_HU_VAL(myBorderData, 0, 0) =
                STARPU_SWE_HUV_MATRIX_GET_HU_VAL(myBlockData, 0, 0);
        STARPU_SWE_HUV_MATRIX_GET_HV_VAL(myBorderData, 0, 0) =
                STARPU_SWE_HUV_MATRIX_GET_HV_VAL(myBlockData, 0, 0);

        STARPU_SWE_HUV_MATRIX_GET_H_VAL(myBorderData, nx + 1, 0) =
                STARPU_SWE_HUV_MATRIX_GET_H_VAL(myBlockData, nx - 1, 0);
        STARPU_SWE_HUV_MATRIX_GET_HU_VAL(myBorderData, nx + 1, 0) =
                STARPU_SWE_HUV_MATRIX_GET_HU_VAL(myBlockData, nx - 1, 0);
        STARPU_SWE_HUV_MATRIX_GET_HV_VAL(myBorderData, nx + 1, 0) =
                STARPU_SWE_HUV_MATRIX_GET_HV_VAL(myBlockData, nx - 1, 0);
    }
    if (side == BND_BOTTOM) {
        STARPU_SWE_HUV_MATRIX_GET_H_VAL(myBorderData, 0, 0) =
                STARPU_SWE_HUV_MATRIX_GET_H_VAL(myBlockData, 0, ny - 1);
        STARPU_SWE_HUV_MATRIX_GET_HU_VAL(myBorderData, 0, 0) =
                STARPU_SWE_HUV_MATRIX_GET_HU_VAL(myBlockData, 0, ny - 1);
        STARPU_SWE_HUV_MATRIX_GET_HV_VAL(myBorderData, 0, 0) =
                STARPU_SWE_HUV_MATRIX_GET_HV_VAL(myBlockData, 0, ny - 1);

        STARPU_SWE_HUV_MATRIX_GET_H_VAL(myBorderData, nx + 1, 0) =
                STARPU_SWE_HUV_MATRIX_GET_H_VAL(myBlockData, nx - 1, ny - 1);
        STARPU_SWE_HUV_MATRIX_GET_HU_VAL(myBorderData, nx + 1, 0) =
                STARPU_SWE_HUV_MATRIX_GET_HU_VAL(myBlockData, nx - 1, ny - 1);
        STARPU_SWE_HUV_MATRIX_GET_HV_VAL(myBorderData, nx + 1, 0) =
                STARPU_SWE_HUV_MATRIX_GET_HV_VAL(myBlockData, nx - 1, ny - 1);
    }

#ifdef DBG
    cout << "Set CONNECT boundary conditions in main memory " << endl << flush;
#endif
}

starpu_codelet SWECodelets::updateGhostLayers = []() noexcept {
    starpu_codelet codelet = {};
    codelet.where = STARPU_CPU;
#ifdef ENABLE_CUDA
    codelet.where |= STARPU_CUDA;
    codelet.cuda_funcs[0] = &updateGhostLayers_cuda;
#endif
    codelet.cpu_funcs[0] = &updateGhostLayers_cpu;
    codelet.nbuffers = 3;
    codelet.modes[0] = STARPU_W;
    codelet.modes[1] = STARPU_R;
    codelet.modes[2] = STARPU_R;
    return codelet;
}();

void writeResult_cpu(void *buffers[], void *cl_arg) {
    io::StarPUBlockWriter *writer;
    starpu_codelet_unpack_args(cl_arg, &writer);

    const auto huvMatrix = reinterpret_cast<SWE_HUV_Matrix_interface *>(buffers[0]);
    const auto bMatrix = reinterpret_cast<starpu_matrix_interface *>(buffers[1]);
    const auto currentTimestamp = (float *) STARPU_VARIABLE_GET_PTR(buffers[2]);

    writer->writeTimeStep(*huvMatrix, *bMatrix, *currentTimestamp);
    printf("writing time step %f\n", *currentTimestamp);

}

starpu_codelet SWECodelets::resultWriter = []()noexcept {
    starpu_codelet codelet = {};
    codelet.where = STARPU_CPU;
    codelet.cpu_funcs[0] = &writeResult_cpu;
    codelet.nbuffers = 3;
    codelet.modes[0] = STARPU_R;
    codelet.modes[1] = STARPU_R;
    codelet.modes[2] = STARPU_R;
    return codelet;
}();

#if defined(SOLVER_AUGRIE)

#include "solvers/AugRieFun.hpp"

static solver::AugRieFun<float_type> waveSolver;
#endif


void computeNumericalFluxes_cpu(void *buffers[], void *cl_arg) {
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

    float_type dx_inv = pBlock->getDx();
    float_type dy_inv = pBlock->getDy();

    float_type l_maxWaveSpeed = 0;

    const auto nX = STARPU_SWE_HUV_MATRIX_GET_NX(mainBlock);
    const auto nY = STARPU_SWE_HUV_MATRIX_GET_NY(mainBlock);

    memset(STARPU_SWE_HUV_MATRIX_GET_H_PTR(netUpdates), 0, sizeof(float_type) * nX * nY);
    memset(STARPU_SWE_HUV_MATRIX_GET_HU_PTR(netUpdates), 0, sizeof(float_type) * nX * nY);
    memset(STARPU_SWE_HUV_MATRIX_GET_HV_PTR(netUpdates), 0, sizeof(float_type) * nX * nY);
#ifdef VECTORIZE
#pragma omp simd
#endif
    for (size_t y = 0; y < nY; ++y) {
        float maxEdgeSpeed;
        float_type hNetUpLeft, hNetUpRight;
        float_type huNetUpLeft, huNetUpRight;

        float_type hLeft = STARPU_SWE_HUV_MATRIX_GET_H_VAL(leftGhost, 0, y);
        float_type hRight = STARPU_SWE_HUV_MATRIX_GET_H_VAL(mainBlock, 0, y);
        float_type huLeft = STARPU_SWE_HUV_MATRIX_GET_HU_VAL(leftGhost, 0, y);
        float_type huRight = STARPU_SWE_HUV_MATRIX_GET_HU_VAL(mainBlock, 0, y);
        float_type bLeft = ((float_type *)
                STARPU_MATRIX_GET_PTR(b))[(y + 1) * STARPU_MATRIX_GET_LD(b)];
        float_type bRight = ((float_type *)
                STARPU_MATRIX_GET_PTR(b))[(y + 1) * STARPU_MATRIX_GET_LD(b) + 1];
        waveSolver.computeNetUpdates(hLeft, hRight,
                                     huLeft, huRight,
                                     bLeft, bRight,
                                     hNetUpLeft, hNetUpRight,
                                     huNetUpLeft, huNetUpRight,
                                     maxEdgeSpeed);
        STARPU_SWE_HUV_MATRIX_GET_H_VAL(netUpdates, 0, y) += dx_inv * hNetUpRight;
        STARPU_SWE_HUV_MATRIX_GET_HU_VAL(netUpdates, 0, y) += dx_inv * huNetUpRight;

        l_maxWaveSpeed = std::max(l_maxWaveSpeed, maxEdgeSpeed);
    }
#ifdef VECTORIZE
#pragma omp simd
#endif
    for (size_t y = 0; y < nY; ++y) {
        float maxEdgeSpeed;
        float_type hNetUpLeft, hNetUpRight;
        float_type huNetUpLeft, huNetUpRight;

        float_type hLeft = STARPU_SWE_HUV_MATRIX_GET_H_VAL(mainBlock, nX - 1, y);
        float_type hRight = STARPU_SWE_HUV_MATRIX_GET_H_VAL(rightGhost, 0, y);
        float_type huLeft = STARPU_SWE_HUV_MATRIX_GET_HU_VAL(mainBlock, nX - 1, y);
        float_type huRight = STARPU_SWE_HUV_MATRIX_GET_HU_VAL(rightGhost, 0, y);
        float_type bLeft = ((float_type *)
                STARPU_MATRIX_GET_PTR(b))[(y + 1) * STARPU_MATRIX_GET_LD(b) + nX];
        float_type bRight = ((float_type *)
                STARPU_MATRIX_GET_PTR(b))[(y + 1) * STARPU_MATRIX_GET_LD(b) + nX + 1];
        waveSolver.computeNetUpdates(hLeft, hRight,
                                     huLeft, huRight,
                                     bLeft, bRight,
                                     hNetUpLeft, hNetUpRight,
                                     huNetUpLeft, huNetUpRight,
                                     maxEdgeSpeed);
        STARPU_SWE_HUV_MATRIX_GET_H_VAL(netUpdates, nX - 1, y) += dx_inv * hNetUpLeft;
        STARPU_SWE_HUV_MATRIX_GET_HU_VAL(netUpdates, nX - 1, y) += dx_inv * huNetUpLeft;

        l_maxWaveSpeed = std::max(l_maxWaveSpeed, maxEdgeSpeed);
    }

    for (size_t y = 0; y < nY; ++y) {
#ifdef VECTORIZE
#pragma omp simd
#endif
        for (size_t x = 1; x < nX; ++x) {
            float maxEdgeSpeed;
            float_type hNetUpLeft, hNetUpRight;
            float_type huNetUpLeft, huNetUpRight;

            float_type hLeft = STARPU_SWE_HUV_MATRIX_GET_H_VAL(mainBlock, x - 1, y);
            float_type hRight = STARPU_SWE_HUV_MATRIX_GET_H_VAL(mainBlock, x, y);
            float_type huLeft = STARPU_SWE_HUV_MATRIX_GET_HU_VAL(mainBlock, x - 1, y);
            float_type huRight = STARPU_SWE_HUV_MATRIX_GET_HU_VAL(mainBlock, x, y);
            float_type bLeft = ((float_type *)
                    STARPU_MATRIX_GET_PTR(b))[(y + 1) * STARPU_MATRIX_GET_LD(b) + (x)];
            float_type bRight = ((float_type *)
                    STARPU_MATRIX_GET_PTR(b))[(y + 1) * STARPU_MATRIX_GET_LD(b) + (x + 1)];
            waveSolver.computeNetUpdates(hLeft, hRight,
                                         huLeft, huRight,
                                         bLeft, bRight,
                                         hNetUpLeft, hNetUpRight,
                                         huNetUpLeft, huNetUpRight,
                                         maxEdgeSpeed);
            STARPU_SWE_HUV_MATRIX_GET_H_VAL(netUpdates, x - 1, y) += dx_inv * hNetUpLeft;
            STARPU_SWE_HUV_MATRIX_GET_HU_VAL(netUpdates, x - 1, y) += dx_inv * huNetUpLeft;

            STARPU_SWE_HUV_MATRIX_GET_H_VAL(netUpdates, x, y) += dx_inv * hNetUpRight;
            STARPU_SWE_HUV_MATRIX_GET_HU_VAL(netUpdates, x, y) += dx_inv * huNetUpRight;

            l_maxWaveSpeed = std::max(l_maxWaveSpeed, maxEdgeSpeed);
        }
    }
#ifdef VECTORIZE
#pragma omp simd
#endif
    for (size_t x = 0; x < nX; ++x) {
        float maxEdgeSpeed;
        float_type hNetUpUpper, hNetUpLower;
        float_type hvNetUpUpper, hvNetUpLower;

        float_type hUpper = STARPU_SWE_HUV_MATRIX_GET_H_VAL(topGhost, x + 1, 0);
        float_type hLower = STARPU_SWE_HUV_MATRIX_GET_H_VAL(mainBlock, x, 0);
        float_type hvUpper = STARPU_SWE_HUV_MATRIX_GET_HV_VAL(topGhost, x + 1, 0);
        float_type hvLower = STARPU_SWE_HUV_MATRIX_GET_HV_VAL(mainBlock, x, 0);

        float_type bUpper = ((float_type *)
                STARPU_MATRIX_GET_PTR(b))[(0) * STARPU_MATRIX_GET_LD(b) + (x + 1)];
        float_type bLower = ((float_type *)
                STARPU_MATRIX_GET_PTR(b))[(0 + 1) * STARPU_MATRIX_GET_LD(b) + (x + 1)];
        waveSolver.computeNetUpdates(hUpper, hLower,
                                     hvUpper, hvLower,
                                     bUpper, bLower,
                                     hNetUpUpper, hNetUpLower,
                                     hvNetUpUpper, hvNetUpLower,
                                     maxEdgeSpeed);

        STARPU_SWE_HUV_MATRIX_GET_H_VAL(netUpdates, x, 0) += dy_inv * hNetUpLower;
        STARPU_SWE_HUV_MATRIX_GET_HV_VAL(netUpdates, x, 0) += dy_inv * hvNetUpLower;

        l_maxWaveSpeed = std::max(l_maxWaveSpeed, maxEdgeSpeed);
    }
#ifdef VECTORIZE
#pragma omp simd
#endif
    for (size_t x = 0; x < nX; ++x) {
        float maxEdgeSpeed;
        float_type hNetUpUpper, hNetUpLower;
        float_type hvNetUpUpper, hvNetUpLower;

        float_type hUpper = STARPU_SWE_HUV_MATRIX_GET_H_VAL(mainBlock, x, nY - 1);
        float_type hLower = STARPU_SWE_HUV_MATRIX_GET_H_VAL(bottomGhost, x + 1, 0);
        float_type hvUpper = STARPU_SWE_HUV_MATRIX_GET_HV_VAL(mainBlock, x, nY - 1);
        float_type hvLower = STARPU_SWE_HUV_MATRIX_GET_HV_VAL(bottomGhost, x + 1, 0);

        float_type bUpper = ((float_type *)
                STARPU_MATRIX_GET_PTR(b))[(nY) * STARPU_MATRIX_GET_LD(b) + (x + 1)];
        float_type bLower = ((float_type *)
                STARPU_MATRIX_GET_PTR(b))[(nY + 1) * STARPU_MATRIX_GET_LD(b) + (x + 1)];
        waveSolver.computeNetUpdates(hUpper, hLower,
                                     hvUpper, hvLower,
                                     bUpper, bLower,
                                     hNetUpUpper, hNetUpLower,
                                     hvNetUpUpper, hvNetUpLower,
                                     maxEdgeSpeed);

        STARPU_SWE_HUV_MATRIX_GET_H_VAL(netUpdates, x, nY - 1) += dy_inv * hNetUpUpper;
        STARPU_SWE_HUV_MATRIX_GET_HV_VAL(netUpdates, x, nY - 1) += dy_inv * hvNetUpUpper;

        l_maxWaveSpeed = std::max(l_maxWaveSpeed, maxEdgeSpeed);
    }
    for (size_t y = 1; y < nY; ++y) {
#ifdef VECTORIZE
#pragma omp simd
#endif
        for (size_t x = 0; x < nX; ++x) {
            float maxEdgeSpeed;
            float_type hNetUpUpper, hNetUpLower;
            float_type hvNetUpUpper, hvNetUpLower;

            float_type hUpper = STARPU_SWE_HUV_MATRIX_GET_H_VAL(mainBlock, x, y - 1);
            float_type hLower = STARPU_SWE_HUV_MATRIX_GET_H_VAL(mainBlock, x, y);
            float_type hvUpper = STARPU_SWE_HUV_MATRIX_GET_HV_VAL(mainBlock, x, y - 1);
            float_type hvLower = STARPU_SWE_HUV_MATRIX_GET_HV_VAL(mainBlock, x, y);

            float_type bUpper = ((float_type *)
                    STARPU_MATRIX_GET_PTR(b))[(y) * STARPU_MATRIX_GET_LD(b) + (x + 1)];
            float_type bLower = ((float_type *)
                    STARPU_MATRIX_GET_PTR(b))[(y + 1) * STARPU_MATRIX_GET_LD(b) + (x + 1)];
            waveSolver.computeNetUpdates(hUpper, hLower,
                                         hvUpper, hvLower,
                                         bUpper, bLower,
                                         hNetUpUpper, hNetUpLower,
                                         hvNetUpUpper, hvNetUpLower,
                                         maxEdgeSpeed);

            STARPU_SWE_HUV_MATRIX_GET_H_VAL(netUpdates, x, y - 1) += dy_inv * hNetUpUpper;
            STARPU_SWE_HUV_MATRIX_GET_HV_VAL(netUpdates, x, y - 1) += dy_inv * hvNetUpUpper;
            STARPU_SWE_HUV_MATRIX_GET_H_VAL(netUpdates, x, y) += dy_inv * hNetUpLower;
            STARPU_SWE_HUV_MATRIX_GET_HV_VAL(netUpdates, x, y) += dy_inv * hvNetUpLower;

            l_maxWaveSpeed = std::max(l_maxWaveSpeed, maxEdgeSpeed);
        }
    }
    if (l_maxWaveSpeed > 0.00001) {
        //compute the time step width
        //CFL-Codition
        //(max. wave speed) * dt / dx < .5
        // => dt = .5 * dx/(max wave speed)

        *maxTimestep = std::min(pBlock->getDx() / l_maxWaveSpeed, pBlock->getDy() / l_maxWaveSpeed);

        // reduce maximum time step size by "safety factor"
        *maxTimestep *= SWECodelets::CFL_NUMBER; //CFL-number = .5
    } else
        //might happen in dry cells
        *maxTimestep = std::numeric_limits<float>::max();
}

starpu_codelet SWECodelets::computeNumericalFluxes = []()noexcept {
    starpu_codelet codelet = {};
/*    codelet.where = STARPU_CPU;
    codelet.cpu_funcs[0] = &computeNumericalFluxes_cpu;*/
#ifdef ENABLE_CUDA
    codelet.where |= STARPU_CUDA;
    codelet.cuda_funcs[0] = &computeNumericalFluxes_cuda;
#endif

    codelet.nbuffers = 8;
    codelet.modes[0] = STARPU_R;
    codelet.modes[1] = STARPU_R;
    codelet.modes[2] = STARPU_R;
    codelet.modes[3] = STARPU_R;
    codelet.modes[4] = STARPU_R;
    codelet.modes[5] = STARPU_R;
    codelet.modes[6] = STARPU_W;
    codelet.modes[7] = STARPU_REDUX;
    return codelet;
}();


void variableMin_cpu(void *buffers[], void *cl_args) {
    float *a = (float *) STARPU_VARIABLE_GET_PTR(buffers[0]);
    float *b = (float *) STARPU_VARIABLE_GET_PTR(buffers[1]);
    *a = std::min(*a, *b);
}


starpu_codelet SWECodelets::variableMin = []()noexcept {
    starpu_codelet codelet = {};
    codelet.where = STARPU_CPU;
    codelet.cpu_funcs[0] = &variableMin_cpu;
#ifdef ENABLE_CUDA
    codelet.where |= STARPU_CUDA;
    codelet.cuda_funcs[0] = &variableMin_cuda;
#endif
    codelet.nbuffers = 2;
    codelet.modes[0] = STARPU_RW;
    codelet.modes[1] = STARPU_R;
    return codelet;
}();

void variableSetInf_cpu(void *buffers[], void *cl_args) {
    float *value = (float *) STARPU_VARIABLE_GET_PTR(buffers[0]);
    *value = std::numeric_limits<float>::infinity();
}

starpu_codelet SWECodelets::variableSetInf = []()noexcept {
    starpu_codelet codelet = {};
    codelet.where = STARPU_CPU;
#ifdef ENABLE_CUDA
    codelet.where |= STARPU_CUDA;
    codelet.cuda_funcs[0] = &variableSetInf_cuda;
#endif
    codelet.cpu_funcs[0] = &variableSetInf_cpu;
    codelet.nbuffers = 1;
    codelet.modes[0] = STARPU_W;
    return codelet;
}();


void updateUnkowns_cpu(void *buffers[], void *cl_args) {
    const SWE_StarPU_Block *pBlock;
    starpu_codelet_unpack_args(cl_args, &pBlock);

    const auto myBlock = buffers[0];
    const auto updates = buffers[1];
    const auto dt = (const float *) STARPU_VARIABLE_GET_PTR(buffers[2]);

    const auto nX = STARPU_SWE_HUV_MATRIX_GET_NX(myBlock);
    const auto nY = STARPU_SWE_HUV_MATRIX_GET_NY(myBlock);

    for (size_t y = 0; y < nY; ++y) {
#pragma omp simd
        for (size_t x = 0; x < nX; ++x) {
            STARPU_SWE_HUV_MATRIX_GET_H_VAL(myBlock, x, y) -= *dt * STARPU_SWE_HUV_MATRIX_GET_H_VAL(updates, x, y);
            STARPU_SWE_HUV_MATRIX_GET_HU_VAL(myBlock, x, y) -= *dt * STARPU_SWE_HUV_MATRIX_GET_HU_VAL(updates, x, y);
            STARPU_SWE_HUV_MATRIX_GET_HV_VAL(myBlock, x, y) -= *dt * STARPU_SWE_HUV_MATRIX_GET_HV_VAL(updates, x, y);

            if (STARPU_SWE_HUV_MATRIX_GET_H_VAL(myBlock, x, y) < SWECodelets::DRY_LIMIT) {
                STARPU_SWE_HUV_MATRIX_GET_H_VAL(myBlock, x, y) =
                STARPU_SWE_HUV_MATRIX_GET_HU_VAL(myBlock, x, y) =
                STARPU_SWE_HUV_MATRIX_GET_HV_VAL(myBlock, x, y) = 0;
            }

        }
    }
}

starpu_codelet SWECodelets::updateUnknowns = []() {
    starpu_codelet codelet = {};
/*    codelet.where = STARPU_CPU;
    codelet.cpu_funcs[0] = &updateUnkowns_cpu;*/
#ifdef ENABLE_CUDA
    codelet.where |= STARPU_CUDA;
    codelet.cuda_funcs[0] = &updateUnknowns_cuda;
#endif

    codelet.nbuffers = 3;
    codelet.modes[0] = STARPU_RW;
    codelet.modes[1] = STARPU_R;
    codelet.modes[2] = STARPU_R;
    return codelet;
}();

void incrementTime_cpu(void *buffers[], void *cl_args) {

    const auto currentTime = (float *) STARPU_VARIABLE_GET_PTR(buffers[0]);
    const auto timestep = (const float *) STARPU_VARIABLE_GET_PTR(buffers[1]);
    const auto nextTimestampToWrite = (float *) STARPU_VARIABLE_GET_PTR(buffers[2]);
    SWE_StarPU_Sim *pSim;
    std::vector<float> *checkpoints;
    starpu_codelet_unpack_args(cl_args, &pSim, &checkpoints);
    *currentTime += *timestep;
    std::cout << "t: "<<*currentTime << '\n';
    if (*nextTimestampToWrite <= *currentTime) {
        pSim->writeTimeStep();
        auto findIt = std::find_if(checkpoints->cbegin(), checkpoints->cend(),
                                   [&](const float test) {
                                       return test > *currentTime;
                                   });
        if (findIt == checkpoints->cend()) {
            //We are done, we are at the last time-step to write
            return;
        }
        //continue
        *nextTimestampToWrite = *findIt;

    }
    pSim->runTimestep();


}

starpu_codelet SWECodelets::incrementTime = []() {
    starpu_codelet codelet = {};
    codelet.where = STARPU_CPU;
    codelet.cpu_funcs[0] = &incrementTime_cpu;
    codelet.nbuffers = 3;
    codelet.modes[0] = STARPU_RW;
    codelet.modes[1] = STARPU_R;
    codelet.modes[2] = STARPU_RW;
    return codelet;
}();