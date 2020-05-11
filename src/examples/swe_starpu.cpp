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


#include "swe-starpu/SWE_StarPU_Block.h"

#include "swe-starpu/testkernels.cuh"


static void test_cpu_func(void *buffers[], void *_args) {
    auto *factor = (float *) _args;
    /* length of the vector */
    unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
    unsigned ny = STARPU_MATRIX_GET_NY(buffers[0]);
    /* local copy of the vector pointer */
    unsigned row_stride = STARPU_MATRIX_GET_LD(buffers[0]);
    auto *val = (float *) STARPU_MATRIX_GET_PTR(buffers[0]);

    for (auto y = 0; y < ny; ++y) {
        for(auto x = 0; x < nx;++x) {
            val[y*row_stride+x] *= *factor;
        }
    }
}




static starpu_codelet test_codelet = []() {
    //Only C++20 has designated initializers
    starpu_codelet codelet = {};
    codelet.where = STARPU_CPU;
    codelet.nbuffers = 1;
    codelet.cpu_funcs[0] = test_cpu_func;
    codelet.modes[0] = STARPU_RW;
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


/*
    using Real = float;
    Real* b = nullptr;
    starpu_malloc((void**)&b, sizeof(Real**));
    struct SOA{
        using float_type = Real;
        void* h = nullptr;
        void* hv = nullptr;
        void* hu = nullptr;
        starpu_data_handle_t spu_h = nullptr;
        starpu_data_handle_t spu_hv = nullptr;
        starpu_data_handle_t spu_hu = nullptr;

        SOA() = default;
        explicit SOA(const size_t nX, const size_t nY){
            starpu_malloc(&h, sizeof(float_type)*nX*nY);
            starpu_malloc(&hu, sizeof(float_type)*nX*nY);
            starpu_malloc(&hv, sizeof(float_type)*nX*nY);

            starpu_matrix_data_register(&spu_h, STARPU_MAIN_RAM, (uintptr_t) h, nX, nX, nY,
                                        sizeof(float_type));
            starpu_matrix_data_register(&spu_hu, STARPU_MAIN_RAM, (uintptr_t) hu, nX, nX, nY,
                                        sizeof(float_type));
            starpu_matrix_data_register(&spu_hv, STARPU_MAIN_RAM, (uintptr_t) hv, nX, nX, nY,
                                        sizeof(float_type));
        }

        ~SOA(){
            if(spu_h) {
                starpu_data_unregister(spu_h);
            }
            if(spu_hu) {
                starpu_data_unregister(spu_hu);
            }
            if(spu_hv) {
                starpu_data_unregister(spu_hv);
            }
            if(h) {
                starpu_free(h);
            }
            if(hu) {
                starpu_free(hu);
            }
            if(hv) {
                starpu_free(hv);
            }
        }
    };

    constexpr uint8_t NLAYERS = 2; //Number of Data copies we have locally. At leas two to have explicit read/write regions
    SOA workingData[NLAYERS];

    constexpr uint32_t nBlocksX = 3;
    constexpr uint32_t nBlocksY = 3;


    for(auto i = 0; i < NLAYERS;++i) {
        workingData[i] = SOA(l_nX+1,l_nY+1);
        //Divide X-Wise:
        starpu_data_filter rowFilter = {};
        rowFilter.filter_func = starpu_matrix_filter_block;
        rowFilter.nchildren = nBlocksY;
        starpu_data_filter colFilter = {};
        colFilter.filter_func = starpu_matrix_filter_vertical_block;
        colFilter.nchildren = nBlocksX;
        starpu_data_map_filters(workingData[i], 2, &rowFilter, &colFilter);
        starpu_data_filter readRowFilter = {};
        readRowFilter.filter_func = starpu_matrix_filter_block_shadow;
        readRowFilter.nchildren = nBlocksY;

        starpu_data_filter readColFilter = {};
        readColFilter.filter_func = starpu_matrix_filter_vertical_block_shadow;
        readColFilter.nchildren = nBlocksX;
        readColFilter.
        starpu_data_map_filters(workingData[i], 2, &rowFilter, &colFilter);
        for (unsigned y = 0; y < nBlocksY; ++y) {
            for (unsigned x = 0; x < nBlocksX; ++x) {
                auto blockHandle = starpu_data_get_sub_data(workingData[i], 2, y, x);

            }
        }
    }





    starpu_task_wait_for_all();
    starpu_data_unregister(dataHandle);
    starpu_free(myMatrix);
    */
    starpu_shutdown();
    return 0;
}
