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

    auto l_dX = (l_scenario.getBoundaryPos(BND_RIGHT) - l_scenario.getBoundaryPos(BND_LEFT) )/l_nX;
    auto l_dY = (l_scenario.getBoundaryPos(BND_TOP) - l_scenario.getBoundaryPos(BND_BOTTOM) )/l_nY;

    //! time when the simulation ends.
    float l_endSimulation = l_scenario.endSimulation();

    //! checkpoints when output files are written.
    float* l_checkPoints = new float[l_numberOfCheckPoints+1];

    // compute the checkpoints in time
    for(int cp = 0; cp <= l_numberOfCheckPoints; cp++) {
        l_checkPoints[cp] = cp*(l_endSimulation/l_numberOfCheckPoints);
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

    SWE_StarPU_Block l_block(l_nX,l_nY,l_dX,l_dY);



    auto l_fileName = generateBaseFileName(l_baseName,0,0);


    l_block.initScenario(0,0,l_scenario);
    l_block.register_starpu();
    auto l_writer = std::make_shared<io::StarPUBlockWriter>(
            l_fileName,
            l_nX,l_nY,
            l_dX,l_dY,
            0,0
            );



    auto side = BND_LEFT;
    auto blockptr = &l_block;
    starpu_task_insert(&SWECodelets::updateGhostLayers,
            STARPU_VALUE,&side,sizeof(side),
            STARPU_VALUE,&blockptr, sizeof(blockptr),
            STARPU_W,l_block.boundaryData[side].starpuHandle(),
            STARPU_R,l_block.huvData().starpuHandle(),
            STARPU_R,l_block.huvData().starpuHandle(),
            0);
    side = BND_RIGHT;
    starpu_task_insert(&SWECodelets::updateGhostLayers,
                       STARPU_VALUE,&side,sizeof(side),
                       STARPU_VALUE,&blockptr, sizeof(blockptr),
                       STARPU_W,l_block.boundaryData[side].starpuHandle(),
                       STARPU_R,l_block.huvData().starpuHandle(),
                       STARPU_R,l_block.huvData().starpuHandle(),
                       0);
    side = BND_BOTTOM;
    starpu_task_insert(&SWECodelets::updateGhostLayers,
                       STARPU_VALUE,&side,sizeof(side),
                       STARPU_VALUE,&blockptr, sizeof(blockptr),
                       STARPU_W,l_block.boundaryData[side].starpuHandle(),
                       STARPU_R,l_block.huvData().starpuHandle(),
                       STARPU_R,l_block.huvData().starpuHandle(),
                       0);
    side = BND_TOP;
    starpu_task_insert(&SWECodelets::updateGhostLayers,
                       STARPU_VALUE,&side,sizeof(side),
                       STARPU_VALUE,&blockptr, sizeof(blockptr),
                       STARPU_W,l_block.boundaryData[side].starpuHandle(),
                       STARPU_R,l_block.huvData().starpuHandle(),
                       STARPU_R,l_block.huvData().starpuHandle(),
                       0);

    const auto writerPtr =  l_writer.get();
    float timestamp = 0;
    starpu_task_insert(&SWECodelets::resultWriter,
            STARPU_VALUE, &writerPtr,sizeof(writerPtr),
            STARPU_VALUE, &timestamp, sizeof(timestamp),
            STARPU_R, l_block.huvData().starpuHandle(),
            STARPU_R, l_block.bStarpuHandle(),
            0);
    timestamp = 1;
    starpu_task_insert(&SWECodelets::resultWriter,
                       STARPU_VALUE, &writerPtr,sizeof(writerPtr),
                       STARPU_VALUE, &timestamp, sizeof(timestamp),
                       STARPU_R, l_block.huvData().starpuHandle(),
                       STARPU_R, l_block.bStarpuHandle(),
                       0);
    l_block.starpu_unregister();
    starpu_shutdown();
    return 0;
}
