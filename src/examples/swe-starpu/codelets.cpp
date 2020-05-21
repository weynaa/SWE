
#include "codelets.h"
#include "SWE_StarPU_Block.h"
#include "writer/StarPUBlockWriter.h"
#include <vector>
#include <algorithm>
#include <limits>

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
            break;
        }
        case CONNECT:
        case PASSIVE:
            break;
        default:
            assert(false);
            break;
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


    if (thisBlock->boundary[side] == CONNECT) {
        auto myNeighbourData = buffers[2];
        const auto neighbourNX = STARPU_SWE_HUV_MATRIX_GET_NX(myNeighbourData);
        const auto neighbourNY = STARPU_SWE_HUV_MATRIX_GET_NY(myNeighbourData);
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
}

starpu_codelet SWECodelets::updateGhostLayers = []() noexcept {
    starpu_codelet codelet = {};
    codelet.where = STARPU_CPU;
    codelet.cpu_funcs[0] = &updateGhostLayers_cpu;
    codelet.nbuffers = 3;
    codelet.modes[0] = STARPU_W;
    codelet.modes[1] = STARPU_R;
    codelet.modes[2] = STARPU_R;
    return codelet;
}();

void writeResult_cpu(void *buffers[], void *cl_arg) {
    io::StarPUBlockWriter *writer;
    std::vector<float> *checkpoints = {};
    starpu_codelet_unpack_args(cl_arg, &writer, &checkpoints);

    const auto huvMatrix = reinterpret_cast<SWE_HUV_Matrix_interface *>(buffers[0]);
    const auto bMatrix = reinterpret_cast<starpu_matrix_interface *>(buffers[1]);
    const auto currentTimestamp = (float *) STARPU_VARIABLE_GET_PTR(buffers[2]);
    const auto nextTimestampToWrite = (float *) STARPU_VARIABLE_GET_PTR(buffers[2]);

    if (nextTimestampToWrite <= currentTimestamp) {
        auto findNext = std::find_if(checkpoints->begin(), checkpoints->end(),
                                     [&](const float test) -> bool {
                                         return test > *nextTimestampToWrite;
                                     });
        if (findNext != checkpoints->end()) {
            *nextTimestampToWrite = *findNext;
        } else {
            *nextTimestampToWrite = std::numeric_limits<float>::infinity();
        }
        writer->writeTimeStep(*huvMatrix, *bMatrix, *currentTimestamp);
    }

}

starpu_codelet SWECodelets::resultWriter = []()noexcept {
    starpu_codelet codelet = {};
    codelet.where = STARPU_CPU;
    codelet.cpu_funcs[0] = &writeResult_cpu;
    codelet.nbuffers = 4;
    codelet.modes[0] = STARPU_R;
    codelet.modes[1] = STARPU_R;
    codelet.modes[2] = STARPU_R;
    codelet.modes[3] = STARPU_RW;
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

    for (size_t y = 0; y < nY; ++y) {
        for (size_t x = 1; x < nX + 2; ++x) {

            const auto leftX = x == 1 ? 0 : x - 2;
            const auto rightX = x == nX + 1 ? 0 : x - 1;
            auto &leftblock = x == 1 ? leftGhost : mainBlock;
            auto &rightBlock = x == nX + 1 ? rightGhost : mainBlock;

            float_type maxEdgeSpeed;
            float_type hNetUpLeft, hNetUpRight;
            float_type huNetUpLeft, huNetUpRight;

            float_type hLeft = STARPU_SWE_HUV_MATRIX_GET_H_VAL(leftblock, leftX, y);
            float_type hRight = STARPU_SWE_HUV_MATRIX_GET_H_VAL(rightBlock, rightX, y);
            float_type huLeft = STARPU_SWE_HUV_MATRIX_GET_HU_VAL(leftblock, leftX, y);
            float_type huRight = STARPU_SWE_HUV_MATRIX_GET_HU_VAL(rightBlock, rightX, y);
            float_type bLeft = ((float_type *)
                    STARPU_MATRIX_GET_PTR(b))[(y + 1) * STARPU_MATRIX_GET_LD(b) + (x)];
            float_type bRight = ((float_type *)
                    STARPU_MATRIX_GET_PTR(b))[(y + 1) * STARPU_MATRIX_GET_LD(b) + (x+1)];
            waveSolver.computeNetUpdates(hLeft, hRight,
                                         huLeft, huRight,
                                         bLeft, bRight,
                                         hNetUpLeft, hNetUpRight,
                                         huNetUpLeft, huNetUpRight,
                                         maxEdgeSpeed);
            if (x != 1) {
                STARPU_SWE_HUV_MATRIX_GET_H_VAL(netUpdates, leftX, y) += dx_inv * hNetUpLeft;
                STARPU_SWE_HUV_MATRIX_GET_HU_VAL(netUpdates, leftX, y) += dx_inv * huNetUpLeft;
            }
            if (x != nX + 1) {
                STARPU_SWE_HUV_MATRIX_GET_H_VAL(netUpdates, rightX, y) += dx_inv * hNetUpRight;
                STARPU_SWE_HUV_MATRIX_GET_HU_VAL(netUpdates, rightX, y) += dx_inv * huNetUpRight;
            }
            l_maxWaveSpeed = std::max(l_maxWaveSpeed, maxEdgeSpeed);
        }
    }
    for (size_t y = 1; y < nY + 1; ++y) {
        for (size_t x = 0; x < nX; ++x) {

            const auto lowerY = y == 1 ? 0 : y - 2;
            const auto upperY = y == nY + 1 ? 0 : y - 1;
            const auto xOfs = (y == 1 || y == nY + 1) ? x + 1 : x;
            const auto uppperBlock = y == 0 ? topGhost : mainBlock;
            const auto lowerBlock = y == nY + 1 ? bottomGhost : mainBlock;

            float_type maxEdgeSpeed;
            float_type hNetUpUpper, hNetUpLower;
            float_type hvNetUpUpper, hvNetUpLower;

            float_type hUpper = STARPU_SWE_HUV_MATRIX_GET_H_VAL(uppperBlock, xOfs, upperY);
            float_type hLower = STARPU_SWE_HUV_MATRIX_GET_H_VAL(lowerBlock, xOfs, lowerY);
            float_type hvUpper = STARPU_SWE_HUV_MATRIX_GET_HV_VAL(mainBlock, xOfs, upperY);
            float_type hvLower = STARPU_SWE_HUV_MATRIX_GET_HV_VAL(mainBlock, xOfs, lowerY);
            float_type bUpper = ((float_type *)
                    STARPU_MATRIX_GET_PTR(b))[(y) * STARPU_MATRIX_GET_LD(b) + (x+1)];
            float_type bLower = ((float_type *)
                    STARPU_MATRIX_GET_PTR(b))[(y + 1) * STARPU_MATRIX_GET_LD(b) + (x+1)];
            waveSolver.computeNetUpdates(hUpper, hLower,
                                         hvUpper, hvLower,
                                         bUpper, bLower,
                                         hNetUpUpper, hNetUpLower,
                                         hvNetUpUpper, hvNetUpLower,
                                         maxEdgeSpeed);
            if (y != 1) {
                STARPU_SWE_HUV_MATRIX_GET_H_VAL(netUpdates, xOfs, upperY) += dy_inv * hNetUpUpper;
                STARPU_SWE_HUV_MATRIX_GET_HV_VAL(netUpdates, xOfs, upperY) += dy_inv * hvNetUpUpper;
            }
            if (y != nY + 1) {
                STARPU_SWE_HUV_MATRIX_GET_H_VAL(netUpdates, xOfs, upperY) += dy_inv * hNetUpLower;
                STARPU_SWE_HUV_MATRIX_GET_HV_VAL(netUpdates, xOfs, upperY) += dy_inv * hvNetUpLower;
            }

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
        *maxTimestep *= (float) .4; //CFL-number = .5
    } else
        //might happen in dry cells
        *maxTimestep = std::numeric_limits<float>::max();
}

starpu_codelet SWECodelets::computeNumericalFluxes = []()noexcept {
    starpu_codelet codelet = {};
    codelet.where = STARPU_CPU;
    codelet.cpu_funcs[0] = &computeNumericalFluxes_cpu;
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
    float *b = (float *) STARPU_VARIABLE_GET_PTR(buffers[0]);
    *a = std::min(*a, *b);
}


starpu_codelet SWECodelets::variableMin = []()noexcept {
    starpu_codelet codelet = {};
    codelet.where = STARPU_CPU;
    codelet.cpu_funcs[0] = &variableMin_cpu;
    codelet.nbuffers = 2;
    codelet.modes[0] = STARPU_RW;
    codelet.modes[1] = STARPU_R;
    return codelet;
}();

void variableSetInf_cpu(void *buffers[], void *cl_args) {
    float *value = (float *) STARPU_VARIABLE_GET_PTR(buffers[0]);
    *value = std::numeric_limits<float>::max();
}

starpu_codelet SWECodelets::variableSetInf = []()noexcept {
    starpu_codelet codelet = {};
    codelet.where = STARPU_CPU;
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
    const auto dt = (const float*) STARPU_VARIABLE_GET_PTR(buffers[2]);

    const auto nX = STARPU_SWE_HUV_MATRIX_GET_NX(myBlock);
    const auto nY = STARPU_SWE_HUV_MATRIX_GET_NY(myBlock);

    for (size_t y = 0; y < nY; ++y) {
        for (size_t x = 0; x < nX; ++x) {
            STARPU_SWE_HUV_MATRIX_GET_H_VAL(myBlock,x,y) -=*dt * STARPU_SWE_HUV_MATRIX_GET_H_VAL(updates,x,y);
            STARPU_SWE_HUV_MATRIX_GET_HU_VAL(myBlock,x,y) -=*dt * STARPU_SWE_HUV_MATRIX_GET_HU_VAL(updates,x,y);
            STARPU_SWE_HUV_MATRIX_GET_HV_VAL(myBlock,x,y) -=*dt * STARPU_SWE_HUV_MATRIX_GET_HV_VAL(updates,x,y);

            STARPU_SWE_HUV_MATRIX_GET_H_VAL(updates,x,y) = 0;
            STARPU_SWE_HUV_MATRIX_GET_H_VAL(updates,x,y) = 0;
            STARPU_SWE_HUV_MATRIX_GET_H_VAL(updates,x,y) = 0;

        }
    }

}

starpu_codelet SWECodelets::updateUnknowns = []() {
    starpu_codelet codelet = {};
    codelet.where = STARPU_CPU;
    codelet.cpu_funcs[0] = &updateUnkowns_cpu;
    codelet.nbuffers = 3;
    codelet.modes[0] = STARPU_RW;
    codelet.modes[1] = STARPU_R;
    codelet.modes[2] = STARPU_R;
    return codelet;
}();

void incrementTime_cpu(void *buffers[], void *cl_args) {
    const auto currentTime = (float *) STARPU_VARIABLE_GET_PTR(buffers[0]);
    const auto timestep = (const float *) STARPU_VARIABLE_GET_PTR(buffers[1]);

    *currentTime += *timestep;

}

starpu_codelet SWECodelets::incrementTime = []() {
    starpu_codelet codelet = {};
    codelet.where = STARPU_CPU;
    codelet.cpu_funcs[0] = &incrementTime_cpu;
    codelet.nbuffers = 2;
    codelet.modes[0] = STARPU_RW;
    codelet.modes[1] = STARPU_R;
    return codelet;
}();