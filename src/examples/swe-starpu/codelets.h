
#ifndef SWE_CODELETS_H
#define SWE_CODELETS_H
#include "StarPUCommon.h"

namespace SWECodelets{
    /**
     * Codelet, which updates the Gost Layer for one side of the task
     * Required Paramters:
     *  0: myBlock- SWE_HUV_Matrix of the current blocks-data - READONLY
     *  1: myBoundary - boundary data: SWE_HUV_Matrix -READWRITE
     *  2: myNeighbour optional neighbour data if there are CONNECT boundaries  -READONLY
     *
     *  In contrast to the push version of the swe_simple implementations,
     *  each boundary updates itself by pulling from the neighbouring layer causing less
     *  synchronicity
     */
    static starpu_codelet updateGhostLayers;
}
#endif //SWE_CODELETS_H
