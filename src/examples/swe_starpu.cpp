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
#include "swe-starpu/codelets.h"

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

    //! number of checkpoints for visualization (at each checkpoint in time, an output file is written).
    constexpr int l_numberOfCheckPoints = 20;


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


    starpu_conf conf = {};
    starpu_conf_init(&conf);
    //conf.ncuda=0;
    //conf.nopencl=0;
    auto starpuret = starpu_init(&conf);
    if (starpuret != 0) {
        std::cerr << "Could not initialize StarPU!\n";
        return 1;
    }
    auto l_originX = l_scenario.getBoundaryPos(BND_LEFT);
    auto l_originY = l_scenario.getBoundaryPos(BND_BOTTOM);

    SWE_StarPU_Block l_block(l_nX, l_nY, l_dX, l_dY);


    auto l_fileName = generateBaseFileName(l_baseName, 0, 0);


    l_block.initScenario(0, 0, l_scenario);
    l_block.register_starpu();
    auto l_writer = std::make_shared<io::StarPUBlockWriter>(
            l_fileName,
            l_nX, l_nY,
            l_dX, l_dY,
            0, 0,
            0
    );

    float *timestamp;
    starpu_malloc((void **) &timestamp, sizeof(float));
    *timestamp = 0;
    starpu_data_handle_t spu_timestamp;
    starpu_variable_data_register(&spu_timestamp, STARPU_MAIN_RAM, (uintptr_t) timestamp, sizeof(float));

    float *nextTimestampToWrite;
    starpu_malloc((void **) &nextTimestampToWrite, sizeof(float));
    *nextTimestampToWrite = 0;
    starpu_data_handle_t spu_nextTimestampToWrite;
    starpu_variable_data_register(&spu_nextTimestampToWrite, STARPU_MAIN_RAM, (uintptr_t) nextTimestampToWrite,
                                  sizeof(float));

    const auto writerPtr = l_writer.get();
    const auto pCheckpoints = &l_checkPoints;

    SWE_StarPU_HUV_Allocation updateScratchData(l_nX, l_nY);
    updateScratchData.register_starpu();

    float maxTimestep = std::numeric_limits<float>::max();
    starpu_data_handle_t spu_maxTimestep;
    starpu_variable_data_register(&spu_maxTimestep, STARPU_MAIN_RAM, (uintptr_t) &maxTimestep, sizeof(maxTimestep));
    starpu_data_set_reduction_methods(spu_maxTimestep, &SWECodelets::variableMin, &SWECodelets::variableSetInf);


    starpu_task_insert(&SWECodelets::resultWriter,
                       STARPU_VALUE, &writerPtr, sizeof(writerPtr),
                       STARPU_VALUE, &pCheckpoints, sizeof(pCheckpoints),
                       STARPU_R, l_block.huvData().starpuHandle(),
                       STARPU_R, l_block.bStarpuHandle(),
                       STARPU_R, spu_timestamp,
                       STARPU_RW, spu_nextTimestampToWrite,
                       0);
    for (int i = 0; i < 500; ++i) {
        auto side = BND_LEFT;
        auto blockptr = &l_block;
        starpu_iteration_push(i);
        starpu_task_insert(&SWECodelets::updateGhostLayers,
                           STARPU_VALUE, &side, sizeof(side),
                           STARPU_VALUE, &blockptr, sizeof(blockptr),
                           STARPU_W, l_block.boundaryData[side].starpuHandle(),
                           STARPU_R, l_block.huvData().starpuHandle(),
                           STARPU_R, l_block.huvData().starpuHandle(),
                           0);
        side = BND_RIGHT;
        starpu_task_insert(&SWECodelets::updateGhostLayers,
                           STARPU_VALUE, &side, sizeof(side),
                           STARPU_VALUE, &blockptr, sizeof(blockptr),
                           STARPU_W, l_block.boundaryData[side].starpuHandle(),
                           STARPU_R, l_block.huvData().starpuHandle(),
                           STARPU_R, l_block.huvData().starpuHandle(),
                           0);
        side = BND_BOTTOM;
        starpu_task_insert(&SWECodelets::updateGhostLayers,
                           STARPU_VALUE, &side, sizeof(side),
                           STARPU_VALUE, &blockptr, sizeof(blockptr),
                           STARPU_W, l_block.boundaryData[side].starpuHandle(),
                           STARPU_R, l_block.huvData().starpuHandle(),
                           STARPU_R, l_block.huvData().starpuHandle(),
                           0);
        side = BND_TOP;
        starpu_task_insert(&SWECodelets::updateGhostLayers,
                           STARPU_VALUE, &side, sizeof(side),
                           STARPU_VALUE, &blockptr, sizeof(blockptr),
                           STARPU_W, l_block.boundaryData[side].starpuHandle(),
                           STARPU_R, l_block.huvData().starpuHandle(),
                           STARPU_R, l_block.huvData().starpuHandle(),
                           0);


        starpu_task_insert(&SWECodelets::computeNumericalFluxes,
                           STARPU_VALUE, &blockptr, sizeof(blockptr),
                           STARPU_R, l_block.huvData().starpuHandle(),
                           STARPU_R, l_block.boundaryData[BND_LEFT].starpuHandle(),
                           STARPU_R, l_block.boundaryData[BND_RIGHT].starpuHandle(),
                           STARPU_R, l_block.boundaryData[BND_BOTTOM].starpuHandle(),
                           STARPU_R, l_block.boundaryData[BND_TOP].starpuHandle(),
                           STARPU_R, l_block.bStarpuHandle(),
                           STARPU_W, updateScratchData.starpuHandle(),
                           STARPU_REDUX, spu_maxTimestep,
                           0);
        starpu_task_insert(&SWECodelets::updateUnknowns,
                           STARPU_VALUE, &blockptr, sizeof(blockptr),
                           STARPU_RW, l_block.huvData().starpuHandle(),
                           STARPU_R, updateScratchData.starpuHandle(),
                           STARPU_R, spu_maxTimestep,
                           0);
        starpu_task_insert(&SWECodelets::incrementTime,
                           STARPU_RW, spu_timestamp,
                           STARPU_R, spu_maxTimestep,
                           0);
        starpu_task_insert(&SWECodelets::resultWriter,
                           STARPU_VALUE, &writerPtr, sizeof(writerPtr),
                           STARPU_VALUE, &pCheckpoints, sizeof(pCheckpoints),
                           STARPU_R, l_block.huvData().starpuHandle(),
                           STARPU_R, l_block.bStarpuHandle(),
                           STARPU_R, spu_timestamp,
                           STARPU_RW, spu_nextTimestampToWrite,
                           0);
        starpu_iteration_pop();
    }
    updateScratchData.unregister_starpu();
    l_block.starpu_unregister();
    starpu_data_unregister(spu_nextTimestampToWrite);
    starpu_free(nextTimestampToWrite);
    starpu_data_unregister(spu_timestamp);
    starpu_free(timestamp);
    starpu_shutdown();
    return 0;
}
