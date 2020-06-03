//
// Created by michael on 31.05.20.
//

#ifndef SWE_CODELETS_CL_H
#define SWE_CODELETS_CL_H

void computeNumericalFluxes_opencl(void *buffers[], void *cl_args);

void variableMin_opencl(void* buffers[],void*cl_args);

void variableSetInf_opencl(void* buffers[],void*cl_args);

void updateUnknowns_opencl(void* buffers[], void* cl_args);
#endif //SWE_CODELETS_CL_H
