#ifndef SWE_SWE_HUV_MATRIX_H
#define SWE_SWE_HUV_MATRIX_H
#include "StarPUCommon.h"
using float_type = float;
struct SWE_HUV_Matrix_interface {
    float_type * h;
    float_type * hu;
    float_type * hv;
    size_t nX,nY,ld; //LD==row stride in StarPU convention
};
#define STARPU_SWE_HUV_MATRIX_GET_H_PTR(x) (((SWE_HUV_Matrix_interface*)(x))->h)
#define STARPU_SWE_HUV_MATRIX_GET_HU_PTR(x) (((SWE_HUV_Matrix_interface*)(x))->hu)
#define STARPU_SWE_HUV_MATRIX_GET_HV_PTR(x) (((SWE_HUV_Matrix_interface*)(x))->hv)
#define STARPU_SWE_HUV_MATRIX_GET_NX(x) (((SWE_HUV_Matrix_interface*)(x))->nX)
#define STARPU_SWE_HUV_MATRIX_GET_NY(x) (((SWE_HUV_Matrix_interface*)(x))->nY)
#define STARPU_SWE_HUV_MATRIX_GET_LD(x) (((SWE_HUV_Matrix_interface*)(x))->ld)

void starpu_swe_huv_matrix_register(starpu_data_handle_t *outHandle,unsigned homeNode,
        float * h, float * hu, float * hv,size_t nX, size_t ld, size_t nY);


#endif //SWE_SWE_HUV_MATRIX_H
