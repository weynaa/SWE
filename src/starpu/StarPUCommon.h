//
// Created by michael on 11.05.20.
//

#ifndef SWE_STARPUCOMMON_H
#define SWE_STARPUCOMMON_H
#ifdef ENABLE_STARPU
#ifdef ENABLE_CUDA
#define STARPU_USE_CUDA
#endif

#include <starpu/1.3/starpu.h>
#endif
#endif //SWE_STARPUCOMMON_H
