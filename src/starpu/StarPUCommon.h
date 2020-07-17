#ifndef SWE_STARPUCOMMON_H
#define SWE_STARPUCOMMON_H
#ifdef ENABLE_STARPU
#ifdef ENABLE_CUDA
#define STARPU_USE_CUDA
#endif
#ifdef ENABLE_OPENCL
#define STARPU_USE_OPENCL
#endif
#include <starpu.h>
#include <starpu_heteroprio.h>
#endif
#endif //SWE_STARPUCOMMON_H
