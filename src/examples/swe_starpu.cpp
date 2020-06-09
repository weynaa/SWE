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
#include <memory>
#include "blocks/SWE_Block.hh"

#include "writer/StarPUBlockWriter.h"

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
#include "swe-starpu/SWE_StarPU_Sim.h"


static constexpr size_t BLOCKS_X_DEFAULT = 3;
static constexpr size_t BLOCKS_Y_DEFAULT = 3;

#ifdef ENABLE_OPENCL
struct starpu_opencl_program opencl_programs;
#endif

#include <starpu_heteroprio.h>

//This heteroprio scheduler is a buggy mess.. somehow the starpu_sched_max_priority is 0 and therefore always the same prio is chosen.. also it is way slower instead of faster with it on.
void init_heteroprio(unsigned sched_ctx) {
    // CPU uses 3 buckets
    if (starpu_cpu_worker_get_count())
    {
        starpu_heteroprio_set_nb_prios(0, STARPU_CPU_IDX, 3);
        // It uses direct mapping idx => idx
        unsigned idx;
        for(idx = 0; idx < 3; ++idx)
        {
            starpu_heteroprio_set_mapping(sched_ctx, STARPU_CPU_IDX, idx, idx);
            starpu_heteroprio_set_faster_arch(sched_ctx, STARPU_CPU_IDX, idx);
        }
    }
    // OpenCL is enabled and uses 2 buckets
    starpu_heteroprio_set_nb_prios(sched_ctx, STARPU_CUDA_IDX, 1);
    starpu_heteroprio_set_mapping(sched_ctx, STARPU_CUDA_IDX, 0, 1);
    // For this bucket OpenCL is the fastest
    starpu_heteroprio_set_faster_arch(sched_ctx, STARPU_CUDA_IDX, 1);
    // And CPU is 4 times slower
    starpu_heteroprio_set_arch_slow_factor(sched_ctx, STARPU_CPU_IDX, 1, 18.0f);
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
    args.addOption("blocksX", 's', "Number of blocks in x direction", args.Optional, false);
    args.addOption("blocksY", 't', "Number of blocks in y direction", args.Optional, false);
    args.addOption("checkpoints", 'n', "Number of blocks checkpoints", args.Optional, false);

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

    const auto blocksX = args.getArgument<int>("blocksX", BLOCKS_X_DEFAULT);
    const auto blocksY = args.getArgument<int>("blocksY", BLOCKS_Y_DEFAULT);


    //! number of checkpoints for visualization (at each checkpoint in time, an output file is written).

    const auto l_numberOfCheckPoints = args.getArgument<int>("checkpoints", 1);

    //! l_baseName of the plots.
    std::string l_baseName;

    SWE_RadialDamBreakScenario l_scenario;

    // read command line parameters
    l_nX = args.getArgument<int>("grid-size-x");
    l_nY = args.getArgument<int>("grid-size-y");
    l_baseName = args.getArgument<std::string>("output-basepath");

    auto l_dX = (l_scenario.getBoundaryPos(BND_RIGHT) - l_scenario.getBoundaryPos(BND_LEFT)) / l_nX;
    auto l_dY = (l_scenario.getBoundaryPos(BND_TOP) - l_scenario.getBoundaryPos(BND_BOTTOM)) / l_nY;

    //! time when the simulation ends.
    float l_endSimulation = l_scenario.endSimulation();

    //! checkpoints when output files are written.
    std::vector<float> l_checkPoints = {};
    l_checkPoints.reserve(l_numberOfCheckPoints + 1);

    // compute the checkpoints in time
    for (int cp = 0; cp <= l_numberOfCheckPoints; cp++) {
        l_checkPoints.push_back((float) cp * (l_endSimulation / l_numberOfCheckPoints));
    }


    starpu_conf conf;
    starpu_conf_init(&conf);
    //conf.ncuda=0;
    //conf.nopencl=0;

    conf.sched_policy_name = "heteroprio";
    conf.sched_policy_init = &init_heteroprio;
    auto starpuret = starpu_init(&conf);
    std::cout << "max prio: "<<starpu_sched_get_max_priority()<<'\n';
#ifdef ENABLE_OPENCL
    const auto loadRet = starpu_opencl_load_opencl_from_file("opencl/codelets.cl", &opencl_programs, NULL);
     STARPU_CHECK_RETURN_VALUE(loadRet, "starpu_opencl_load_opencl_from_file");
#endif
    if (starpuret != 0) {
        std::cerr << "Could not initialize StarPU!\n";
        return 1;
    }

    // Init fancy progressbar
    tools::ProgressBar progressBar(l_endSimulation);

    // write the output at time zero
    tools::Logger::logger.printOutputTime((float) 0.);
    progressBar.update(0.);
    {
        SWE_StarPU_Sim sim{
                (size_t) l_nX, (size_t) l_nY,
                (size_t) blocksX, (size_t) blocksY,
                l_dX, l_dY,
                l_baseName,
                l_scenario,
                l_numberOfCheckPoints
        };
        tools::Logger::logger.printStartMessage();
        tools::Logger::logger.initWallClockTime(time(NULL));
        tools::Logger::logger.resetClockToCurrentTime("Cpu");
        sim.launchTaskGraph();
        starpu_task_wait_for_all();
        tools::Logger::logger.updateTime("Cpu");
        tools::Logger::logger.printStatisticsMessage();
        tools::Logger::logger.printTime("Cpu", "CPU time");
        tools::Logger::logger.printWallClockTime(time(NULL));
    }
#ifdef ENABLE_OPENCL
    starpu_opencl_unload_opencl(&opencl_programs);
#endif
    starpu_shutdown();
    return 0;
}
