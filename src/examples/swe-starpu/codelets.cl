#define float_type float

__kernel void updateUnknowns_opencl_kernel(
        __global float_type * mainH,
        __global float_type * mainHu,
        __global float_type * mainHv,
        __global float_type * netUpH,
        __global float_type * netUpHu,
        __global float_type * netUpHv,
        __global float* dt,
        unsigned int nX,
        unsigned int nY
        ){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if(x >= nX || y >= nY){
        return;
    }
    mainH[y*nX+x] -= *dt * netUpH[y*nX+x];
    mainHu[y*nX+x] -= *dt * netUpHu[y*nX+x];
    mainHv[y*nX+x] -= *dt * netUpHv[y*nX+x];

    if(mainH[y*nX+x] < 0.1){
        mainH[y*nX+x] = mainHu[y*nX+x] = mainHv[y*nX+x] = 0;
    }

}