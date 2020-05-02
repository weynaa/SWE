/**
 * @file
 * This file is part of SWE.
 *
 * @author Alexander Breuer (breuera AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Dipl.-Math._Alexander_Breuer)
 *         Michael Bader (bader AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Univ.-Prof._Dr._Michael_Bader)
 *
 * @section LICENSE
 *
 * SWE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SWE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SWE.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * @section DESCRIPTION
 *
 * Basic setting of SWE, which uses a wave propagation solver and an artificial or ASAGI scenario on a single block.
 */

#include <cassert>
#include <cstdlib>
#include <string>
#include <iostream>

#include "blocks/SWE_Block.hh"

#include "writer/Writer.hh"

#ifdef ASAGI
#include "scenarios/SWE_AsagiScenario.hh"
#else

#include "scenarios/SWE_simple_scenarios.hh"

#endif

#ifdef READXML
#include "tools/CXMLConfig.hpp"
#endif

#include "tools/args.hh"
#include "tools/help.hh"
#include "tools/Logger.hh"
#include "tools/ProgressBar.hh"

#define STARPU_USE_CUDA
#define STARPU_USE_MPI
#include <starpu.h>

static void test_cpu_func(void *buffers[], void *_args) {
    float *factor = (float *) _args;
    /* length of the vector */
    unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
    unsigned ny = STARPU_MATRIX_GET_NY(buffers[0]);
    /* local copy of the vector pointer */
    unsigned row_stride = STARPU_MATRIX_GET_LD(buffers[0]);
    float *val = (float *) STARPU_MATRIX_GET_PTR(buffers[0]);

    for (auto y = 0; y < ny; ++y) {
        for(auto x = 0; x < nx;++x) {
            val[y*row_stride+x] *= *factor;
        }
    }
}


static void __global__ test_cuda_kernel(float *val, float factor, size_t nx, size_t ny, size_t stride) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < nx && y < ny) {
        val[y*stride+x] *= factor;
    }
}

static void test_cuda_func(void *buffers[], void *args) {
    float *factor = (float *) args;
    /* length of the vector */
    unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
    unsigned ny = STARPU_MATRIX_GET_NY(buffers[0]);
    unsigned offset = STARPU_MATRIX_GET_OFFSET(buffers[0]);
    unsigned row_stride = STARPU_MATRIX_GET_LD(buffers[0]);
    const auto n = nx*ny;
    /* local copy of the vector pointer */
    float *val = (float *) STARPU_MATRIX_GET_PTR(buffers[0]);
    dim3 threads_per_block = {8,8};
    dim3 nblocks = {(nx + threads_per_block.x - 1) / threads_per_block.x,
                    (ny + threads_per_block.y - 1) / threads_per_block.y};
    test_cuda_kernel<<<nblocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>>(val, *factor, nx,ny,row_stride);
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

static starpu_codelet test_codelet = []() {
    //Only C++20 has designated initializers
    starpu_codelet codelet = {};
    codelet.where = STARPU_CPU | STARPU_CUDA;
    codelet.nbuffers = 1;
    codelet.cpu_funcs[0] = test_cpu_func;
    codelet.modes[0] = STARPU_RW;
    codelet.cuda_funcs[0] = test_cuda_func;
    return codelet;
}();


/**
 * Main program for the simulation on a single SWE_WavePropagationBlock.
 */
int main(int argc, char **argv) {
    /**
     * Initialization.
     */
    // Parse command line parameters
    tools::Args args;
    args.addOption("grid-size-x", 'x', "Number of cells in x direction");
    args.addOption("grid-size-y", 'y', "Number of cells in y direction");
    args.addOption("output-basepath", 'o', "Output base file name");

    tools::Args::Result ret = args.parse(argc, argv);

    switch (ret) {
        case tools::Args::Error:
            return 1;
        case tools::Args::Help:
            return 0;
        default:
            break;
    }

    //! number of grid cells in x- and y-direction.
    int l_nX, l_nY;


    //! l_baseName of the plots.
    std::string l_baseName;

    // read command line parameters
    l_nX = args.getArgument<int>("grid-size-x");
    l_nY = args.getArgument<int>("grid-size-y");
    l_baseName = args.getArgument<std::string>("output-basepath");

    starpu_conf conf = {};
    starpu_conf_init(&conf);
    //conf.ncuda=0;
    //conf.nopencl=0;
    auto starpuret = starpu_init(&conf);
    if (starpuret != 0) {
        std::cerr << "Could not initialize StarPU!\n";
        return 1;
    }

    printf("StarPU workers:\n");
    printf("%d CPU cores\n", starpu_worker_get_count_by_type(STARPU_CPU_WORKER));
    printf("%d CUDA GPUs\n", starpu_worker_get_count_by_type(STARPU_CUDA_WORKER));
    printf("%d OpenCL GPUs\n", starpu_worker_get_count_by_type(STARPU_OPENCL_WORKER));

    const auto grid_Data = (float *) malloc(sizeof(float) * l_nX * l_nY);
    for (int i = 0; i < l_nX*l_nY; ++i) {
        grid_Data[i] = 1.0f;
    }
    starpu_data_handle_t dataHandle;
    starpu_matrix_data_register(&dataHandle, STARPU_MAIN_RAM, (uintptr_t) grid_Data, l_nX, l_nX, l_nY,
                                sizeof(grid_Data[0]));
    const float factor = 2.12116315;

    constexpr int NTASKS = 10;
    starpu_data_filter dataFilter = {};
    dataFilter.filter_func = starpu_matrix_filter_block;
    dataFilter.nchildren = NTASKS;
    starpu_data_partition(dataHandle, &dataFilter);

    for(int i = 0; i < starpu_data_get_nb_children(dataHandle);++i)
    {
        starpu_data_handle_t partHandle = starpu_data_get_sub_data(dataHandle,1,i);
        auto task = starpu_task_create();

        task->synchronous = 1;
        task->cl = &test_codelet;
        task->handles[0] = partHandle;
        task->cl_arg = (void *) &factor;
        task->cl_arg_size = sizeof(factor);

        starpu_task_submit(task);
    }
    for(int i = 0; i < starpu_data_get_nb_children(dataHandle);++i){
        starpu_data_unpartition(dataHandle,i);
    }

    starpu_data_unregister(dataHandle);

    for (int i = 0; i < l_nX*l_nY; ++i) {
        if(grid_Data[i] != factor)
        {
            printf("error");
        }
    }
    starpu_shutdown();
    return 0;
}
