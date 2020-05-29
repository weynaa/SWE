//
// Created by michael on 27.05.20.
//

#ifndef SWE_CODELETS_CUH
#define SWE_CODELETS_CUH


void computeNumericalFluxes_cuda(void *buffers[], void *cl_args);

void variableMin_cuda(void* buffers[],void*cl_args);

void variableSetInf_cuda(void* buffers[],void*cl_args);

void updateUnknowns_cuda(void* buffers[],void*cl_args);


#endif //SWE_CODELETS_CUH
