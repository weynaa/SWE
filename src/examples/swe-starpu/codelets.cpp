/*
#include "codelets.h"
#include "SWE_StarPU_Block.h"
using float_type = SWE_StarPU_Block::float_type;
void updateGhostLayers_cpu(void* buffers[], void* cl_arg){
    SWE_StarPU_Block* thisBlock;
    uint8_t side;
    starpu_codelet_unpack_args(cl_arg, &thisBlock,&side);
    auto h = (float_type *)buffers[0];
    auto hu = (float_type *)buffers[1];
    auto hv = (float_type *)buffers[2];
#ifdef DBG
    cout << "Set simple boundary conditions " << endl << flush;
#endif
    const auto rowStride = thisBlock->rowStride();
    const auto nx = thisBlock->getNx();
    const auto ny = thisBlock->getNy();

    const auto neighbourH = (const float_type*) buffers[3];
    const auto neighbourHu = (const float_type*) buffers[4];
    const auto neighbourHv = (const float_type*) buffers[5];

    switch(thisBlock->boundary[side]) {
        case WALL:
        {
            for(int j=1; j<=ny; j++) {
                h[j*rowStride+0] = h[j*rowStride+1];
                hu[j*rowStride+0] = -hu[j*rowStride+1];
                hv[j*rowStride+0] = hv[j*rowStride+1];
            };
            break;
        }
        case OUTFLOW:
        {
            for(int j=1; j<=ny; j++) {
                h[j*rowStride+0] = h[j*rowStride+1];
                hu[j*rowStride+0] = hu[j*rowStride+1];
                hv[j*rowStride+0] = hv[j*rowStride+1];
            };
            break;
        }
        case CONNECT:
        case PASSIVE:
            break;
        default:
            assert(false);
            break;
    };

    // right boundary
    switch(thisBlock->boundary[BND_RIGHT]) {
        case WALL:
        {
            for(int j=1; j<=ny; j++) {
                h[j*rowStride+nx+1] = h[j*rowStride+nx];
                hu[j*rowStride+nx+1] = -hu[j*rowStride+nx];
                hv[j*rowStride+nx+1] = hv[j*rowStride+nx];
            };
            break;
        }
        case OUTFLOW:
        {
            for(int j=1; j<=ny; j++) {
                h[j*rowStride+nx+1] = h[j*rowStride+nx];
                hu[j*rowStride+nx+1] = hu[j*rowStride+nx];
                hv[j*rowStride+nx+1] = hv[j*rowStride+nx];
            };
            break;
        }
        case CONNECT:
        case PASSIVE:
            break;
        default:
            assert(false);
            break;
    };

    // top boundary
    switch(thisBlock->boundary[BND_TOP]) {
        case WALL:
        {
            for(int i=1; i<=nx; i++) {
                h[i] = h[1*rowStride+i];
                hu[i] = hu[1*rowStride+i];
                hv[i] = -hv[1*rowStride+i];
            };
            break;
        }
        case OUTFLOW:
        {
            for(int i=1; i<=nx; i++) {
                h[i] = h[1*rowStride+i];
                hu[i] = hu[1*rowStride+i];
                hv[i] = hv[1*rowStride+i];
            };
            break;
        }
        case CONNECT:
        case PASSIVE:
            break;
        default:
            assert(false);
            break;
    };

    // bottom boundary
    switch(thisBlock->boundary[BND_BOTTOM]) {
        case WALL:
        {
            for(int i=1; i<=nx; i++) {
                h[(ny+1)*rowStride+i] = h[ny*rowStride+i];
                hu[(ny+1)*rowStride+i] = hu[ny*rowStride+i];
                hv[(ny+1)*rowStride+i] = -hv[ny*rowStride+i];
            };
            break;
        }
        case OUTFLOW:
        {
            for(int i=1; i<=nx; i++) {
                h[(ny+1)*rowStride+i] = h[ny*rowStride+i];
                hu[(ny+1)*rowStride+i] = hu[ny*rowStride+i];
                hv[(ny+1)*rowStride+i] = hv[ny*rowStride+i];
            };
            break;
        }
        case CONNECT:
        case PASSIVE:
            break;
        default:
            assert(false);
            break;
    };

    //Update the 4 corners
    h [0] = h[1*rowStride+1];
    hu[0] = hu[1*rowStride+1];
    hv[0] = hv[1*rowStride+1];

    h [(ny+1)*rowStride] = h [ny*rowStride+1];
    hu[(ny+1)*rowStride] = hu[ny*rowStride+1];
    hv[(ny+1)*rowStride] = hv[ny*rowStride+1];

    h [nx+1] = h [1*rowStride+nx];
    hu[nx+1] = hu[1*rowStride+nx];
    hv[nx+1] = hv[1*rowStride+nx];

    h [(ny+1)*rowStride+nx+1] = h [ny*rowStride+nx];
    hu[(ny+1)*rowStride+nx+1] = hu[ny*rowStride+nx];
    hv[(ny+1)*rowStride+nx+1] = hv[ny*rowStride+nx];

#ifdef DBG
    cout << "Set CONNECT boundary conditions in main memory " << endl << flush;
#endif
    // left boundary
    if (thisBlock->boundary[BND_LEFT] == CONNECT) {
        for(int j=0; j<=ny+1; j++) {
            h[j*rowStride()] = neighbour[BND_LEFT]->h[j];
            hu[j*rowStride()] = neighbour[BND_LEFT]->hu[j];
            hv[j*rowStride()] = neighbour[BND_LEFT]->hv[j];
        };
    };

    // right boundary
    if(boundary[BND_RIGHT] == CONNECT) {
        for(int j=0; j<=ny+1; j++) {
            h[nx+1][j] = neighbour[BND_RIGHT]->h[j];
            hu[nx+1][j] = neighbour[BND_RIGHT]->hu[j];
            hv[nx+1][j] = neighbour[BND_RIGHT]->hv[j];
        };
    };

    // bottom boundary
    if(boundary[BND_BOTTOM] == CONNECT) {
        for(int i=0; i<=nx+1; i++) {
            h[i][0] = neighbour[BND_BOTTOM]->h[i];
            hu[i][0] = neighbour[BND_BOTTOM]->hu[i];
            hv[i][0] = neighbour[BND_BOTTOM]->hv[i];
        };
    };

    // top boundary
    if(boundary[BND_TOP] == CONNECT) {
        for(int i=0; i<=nx+1; i++) {
            h[i][ny+1] = neighbour[BND_TOP]->h[i];
            hu[i][ny+1] = neighbour[BND_TOP]->hu[i];
            hv[i][ny+1] = neighbour[BND_TOP]->hv[i];
        }
    };

#ifdef DBG
    cout << "Synchronize ghost layers (for heterogeneous memory) " << endl << flush;
#endif
    // synchronize the ghost layers (for PASSIVE and CONNECT conditions)
    // with accelerator memory
}

starpu_codelet updateGhostLayers = [](){
    starpu_codelet codelet = {};
    codelet.cpu_funcs[0] = updateGhostLayers_cpu;
    codelet.nbuffers = 6;
    codelet.modes = {STARPU_RW,STARPU_R,STARPU_R, STARPU_R,STARPU_R};
}();

*/
