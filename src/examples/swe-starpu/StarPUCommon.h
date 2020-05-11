//
// Created by michael on 11.05.20.
//

#ifndef SWE_STARPUCOMMON_H
#define SWE_STARPUCOMMON_H

#ifdef ENABLE_CUDA
#define STARPU_USE_CUDA
#endif

#ifdef ENABLE_OPENCL
#define STARPU_USE_OPENCL
#endif

#include <starpu.h>

#endif //SWE_STARPUCOMMON_H
