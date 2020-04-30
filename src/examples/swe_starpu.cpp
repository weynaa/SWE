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

#include <cuda_runtime.h>
#include <starpu.h>

/**
 * Main program for the simulation on a single SWE_WavePropagationBlock.
 */
int main(int argc, char **argv)
{
  /**
   * Initialization.
   */
  // Parse command line parameters
  tools::Args args;
  args.addOption("grid-size-x", 'x', "Number of cells in x direction");
  args.addOption("grid-size-y", 'y', "Number of cells in y direction");
  args.addOption("output-basepath", 'o', "Output base file name");

  tools::Args::Result ret = args.parse(argc, argv);

  switch (ret)
  {
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
  if (starpuret != 0)
  {
    return 1;
  }
  printf("%d CPU cores\n", starpu_worker_get_count_by_type(STARPU_CPU_WORKER));
  printf("%d CUDA GPUs\n", starpu_worker_get_count_by_type(STARPU_CUDA_WORKER));
  printf("%d OpenCL GPUs\n", starpu_worker_get_count_by_type(STARPU_OPENCL_WORKER));
  starpu_shutdown();
  return 0;
}
