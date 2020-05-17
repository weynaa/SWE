
#include "codelets.h"
#include "SWE_StarPU_Block.h"
#include "writer/StarPUBlockWriter.h"

void updateGhostLayers_cpu(void *buffers[], void *cl_arg) {
    const SWE_StarPU_Block *thisBlock;
    BoundaryEdge side;
    starpu_codelet_unpack_args(cl_arg, &side,&thisBlock);
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
    io::StarPUBlockWriter* writer;
    float timestamp;
    starpu_codelet_unpack_args(cl_arg,&writer,&timestamp);

    const auto  huvMatrix = reinterpret_cast<SWE_HUV_Matrix_interface *>(buffers[0]);
    const auto  bMatrix = reinterpret_cast<starpu_matrix_interface *>(buffers[1]);

    writer->writeTimeStep(*huvMatrix,*bMatrix,timestamp);
    
}

starpu_codelet SWECodelets::resultWriter = []()noexcept{
    starpu_codelet codelet = {};
    codelet.where = STARPU_CPU;
    codelet.cpu_funcs[0] = &writeResult_cpu;
    codelet.nbuffers = 2;
    codelet.modes[0] = STARPU_R;
    codelet.modes[1] = STARPU_R;
    return codelet;
}();

