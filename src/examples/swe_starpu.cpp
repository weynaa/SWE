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
#include <mpi.h>

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
#include <starpu_mpi.h>

#include "swe-starpu/testkernels.cuh"


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


int dataDistribution2D(int size, int x, int y)
{
    return ((int)(x/std::sqrt(size)+(y/std::sqrt(size))*std::sqrt(size)))%size;
}

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
    auto starpuret = starpu_mpi_init_conf(&argc,&argv,1,MPI_COMM_WORLD,&conf);
    int rank, worldSize;
    starpu_mpi_comm_rank(MPI_COMM_WORLD,&rank);
    starpu_mpi_comm_size(MPI_COMM_WORLD,&worldSize);


    if (starpuret != 0) {
        std::cerr << "Could not initialize StarPU!\n";
        return 1;
    }

    const auto grid_Data = new float[l_nX * l_nY];
    for (int i = 0; i < l_nX*l_nY; ++i) {
        grid_Data[i] = 1.0f;
    }
    starpu_data_handle_t dataHandle;
    starpu_matrix_data_register(&dataHandle, STARPU_MAIN_RAM, (uintptr_t) grid_Data, l_nX, l_nX, l_nY,
                                sizeof(grid_Data[0]));

    starpu_data_filter dataFilter = {};
    dataFilter.filter_func = starpu_matrix_filter_block;
    dataFilter.nchildren = worldSize;
    starpu_data_partition(dataHandle, &dataFilter);

    starpu_data_handle_t partHandle = starpu_data_get_sub_data(dataHandle,1,rank);
    starpu_mpi_data_register(partHandle,rank,rank);

    const float factor = 2.12116315;

/*    constexpr int NTASKS = 10;
    starpu_data_filter dataFilter = {};
    dataFilter.filter_func = starpu_matrix_filter_block;
    dataFilter.nchildren = NTASKS;
    starpu_data_partition(dataHandle, &dataFilter);*/

    //for(int i = 0; i < starpu_data_get_nb_children(dataHandle);++i)

/*            starpu_data_handle_t partHandle = starpu_data_get_sub_data(dataHandle, 1, i);
            auto task = starpu_task_create();

            task->cl = &test_codelet;
            task->handles[0] = partHandle;
            task->cl_arg = (void *) &factor;
            task->cl_arg_size = sizeof(factor);

            starpu_task_submit(task);*/
            starpu_mpi_task_insert(MPI_COMM_WORLD, &test_codelet,
                    STARPU_VALUE, &factor,sizeof(factor),
                    STARPU_RW, partHandle,
                    0);

    starpu_task_wait_for_all();
/*    for(int i = 0; i < starpu_data_get_nb_children(dataHandle);++i){
        starpu_data_unpartition(dataHandle,i);
    }*/


 /*   for (int i = 0; i < l_nX*l_nY; ++i) {
        if(grid_Data[i] != factor)
        {
            printf("error");
        }
    }*/

    starpu_mpi_shutdown();
    delete[] grid_Data;
    return 0;
}
