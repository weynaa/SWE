#include "codelets_cl.h"
#include <starpu/StarPUCommon.h>
#include "SWE_StarPU_Block.h"
#include <iostream>
#include <cmath>

extern struct starpu_opencl_program opencl_programs;

//OpenCL needs the global grid size to be divisible by local grid size
inline void adjustGlobalDim(size_t & global, const size_t local){
    global = ((global+local-1)/local)*local;
}

void updateUnknowns_opencl(void* buffers[], void* cl_args){
    const SWE_StarPU_Block *pBlock;
    starpu_codelet_unpack_args(cl_args, &pBlock);

    const auto myBlock = buffers[0];
    const auto updates = buffers[1];
    const auto dt = (cl_mem) STARPU_VARIABLE_GET_PTR(buffers[2]);

    const unsigned int nX = STARPU_SWE_HUV_MATRIX_GET_NX(myBlock);
    const unsigned int nY = STARPU_SWE_HUV_MATRIX_GET_NY(myBlock);

    const cl_mem h = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_H_PTR(myBlock);
    const cl_mem hu = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_HU_PTR(myBlock);
    const cl_mem hv = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_HV_PTR(myBlock);

    const cl_mem upH = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_H_PTR(updates);
    const cl_mem upHu = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_HU_PTR(updates);
    const cl_mem upHv = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_HV_PTR(updates);


    int id, devid, err;                   /* OpenCL specific code */
    cl_kernel kernel;                     /* OpenCL specific code */
    cl_command_queue queue;               /* OpenCL specific code */
    cl_event event;                       /* OpenCL specific code */

    id = starpu_worker_get_id();
    devid = starpu_worker_get_devid(id);
    err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_programs,
                                    "updateUnknowns_opencl_kernel", /* Name of the codelet */
                                    devid);
    if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &h);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &hu);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &hv);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &upH);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &upHu);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &upHv);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &dt);
    err |= clSetKernelArg(kernel, 7, sizeof(unsigned int), &nX);
    err |= clSetKernelArg(kernel, 8, sizeof(unsigned int), &nY);
    if (err) STARPU_OPENCL_REPORT_ERROR(err);

    size_t global[2] = {nX,nY};
    size_t local[2];
    size_t s;
    cl_device_id device;
    starpu_opencl_get_device(devid, &device);
    err = clGetKernelWorkGroupInfo (kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), local, &s);
    local[0]=local[1] = std::floor(std::sqrt(local[0]));
    if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    if (local[0] > global[0]) local[0]=global[0];
    if (local[1] > global[1]) local[1]=global[1];
    global[0]= ((global[0]+local[0]-1)/local[0])*local[0];
    global[1]= ((global[1]+local[1]-1)/local[1])*local[1];
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);
    if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    
    starpu_opencl_release_kernel(kernel);
}

//OpenCL is so barebones, it doesn't even have memset lol
void cl_memset(cl_kernel kernel, cl_command_queue queue,int devid, cl_mem memory, const float value, const size_t n, cl_event & event){
    int err = clSetKernelArg(kernel,0,sizeof(cl_mem),&memory);
    err |= clSetKernelArg(kernel,1,sizeof(value),&value);
    err |= clSetKernelArg(kernel,2,sizeof(n),&n);
    if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

    size_t local;
    cl_device_id device;
    starpu_opencl_get_device(devid, &device);
    size_t s;
    err = clGetKernelWorkGroupInfo (kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, &s);
    if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    size_t global = n;
    adjustGlobalDim(global,local);
    err = clEnqueueNDRangeKernel(queue,kernel,1,NULL,&global,&local,0,NULL,&event);
    if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
}

void computeNumericalFluxes_opencl(void *buffers[], void *cl_args){
    const SWE_StarPU_Block *pBlock;
    starpu_codelet_unpack_args(cl_args, &pBlock);

    const auto mainBlock =  buffers[0];
    const auto leftGhost =  buffers[1 + BND_LEFT];
    const auto rightGhost =  buffers[1 + BND_RIGHT];
    const auto bottomGhost =  buffers[1 + BND_BOTTOM];
    const auto topGhost =  buffers[1 + BND_TOP];
    const auto b = (cl_mem) STARPU_MATRIX_GET_PTR(buffers[5]);
    const auto netUpdates =  buffers[6];

    const auto maxTimestep = (cl_mem) STARPU_VARIABLE_GET_PTR(buffers[7]);

    const auto nX = pBlock->getNx();
    const auto nY = pBlock->getNy();

    const auto dX = pBlock->getDx();
    const auto dY = pBlock->getDy();

    const auto dX_inv = 1 / dX;
    const auto dY_inv = 1 / dY;

    const auto hMain = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_H_PTR(mainBlock);
    const auto huMain = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_HU_PTR(mainBlock);
    const auto hvMain = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_HV_PTR(mainBlock);

    const auto hNetUp = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_H_PTR(netUpdates);
    const auto huNetUp = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_HU_PTR(netUpdates);
    const auto hvNetUp = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_HV_PTR(netUpdates);

    int id, devid, err;                   /* OpenCL specific code */
    cl_command_queue queue;               /* OpenCL specific code */
    cl_event hMemset,huMemset,hvMemset;   /* OpenCL specific code */

    id = starpu_worker_get_id();
    devid = starpu_worker_get_devid(id);



    cl_device_id device;
    starpu_opencl_get_device(devid, &device);

    cl_kernel memset_kernel;

    starpu_opencl_load_kernel(&memset_kernel,&queue,&opencl_programs,
                              "memset", /* Name of the codelet */
                              devid);

    cl_memset(memset_kernel,queue,devid,hNetUp,0,nX*nY,hMemset);
    cl_memset(memset_kernel,queue,devid,huNetUp,0,nX*nY,huMemset);
    cl_memset(memset_kernel,queue,devid,hvNetUp,0,nX*nY,hvMemset);

    starpu_opencl_release_kernel(memset_kernel);

    //Boundary Updates
    {
        cl_kernel kernel;
        err = starpu_opencl_load_kernel(&kernel,&queue,&opencl_programs,
                                        "computeNumericalFluxes_border_opencl_kernel",
                                        devid);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

        size_t workGroupSize;
        size_t s;
        err = clGetKernelWorkGroupInfo (kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(workGroupSize), &workGroupSize, &s);

        BoundaryEdge side = BND_LEFT;
        const auto hNeighbour = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_H_PTR(leftGhost);
        const auto huNeighbour = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_HU_PTR(leftGhost);
        const auto hvNeighbour = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_HV_PTR(leftGhost);
        err |= clSetKernelArg(kernel,0,sizeof(cl_mem),&hMain);
        err |= clSetKernelArg(kernel,1,sizeof(cl_mem),&huMain);
        err |= clSetKernelArg(kernel,2,sizeof(cl_mem),&hvMain);
        err |= clSetKernelArg(kernel,3,sizeof(cl_mem),&hNeighbour);
        err |= clSetKernelArg(kernel,4,sizeof(cl_mem),&huNeighbour);
        err |= clSetKernelArg(kernel,5,sizeof(cl_mem),&hvNeighbour);
        err |= clSetKernelArg(kernel,6,sizeof(cl_mem),&b);
        err |= clSetKernelArg(kernel,7,sizeof(cl_mem),&maxTimestep);
        err |= clSetKernelArg(kernel,8,sizeof(cl_mem),&hNetUp);
        err |= clSetKernelArg(kernel,9,sizeof(cl_mem),&huNetUp);
        err |= clSetKernelArg(kernel,10,sizeof(cl_mem),&hvNetUp);
        err |= clSetKernelArg(kernel,11,sizeof(nX),&nX);
        err |= clSetKernelArg(kernel,12,sizeof(nY),&nY);
        err |= clSetKernelArg(kernel,13,sizeof(dX_inv),&dX_inv);
        err |= clSetKernelArg(kernel,14,sizeof(dY_inv),&dY_inv);
        err |= clSetKernelArg(kernel,15,sizeof(dX),&dX);
        err |= clSetKernelArg(kernel,16,sizeof(dY),&dY);
        err |= clSetKernelArg(kernel,17,sizeof(side),&side);
        err |= clSetKernelArg(kernel,18, sizeof(float)*workGroupSize,NULL);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

        size_t global = nY;
        adjustGlobalDim(global,workGroupSize);

        cl_event dependencies[]={hMemset,huMemset,hvMemset};
        cl_event leftEvent,rightEvent,topEvent,botEvent;
        err = clEnqueueNDRangeKernel(queue,kernel,1,NULL,&global,&workGroupSize,3,dependencies,&leftEvent);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);


        const auto hRightNeighbour = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_H_PTR(rightGhost);
        const auto huRightNeighbour = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_HU_PTR(rightGhost);
        const auto hvRightNeighbour = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_HV_PTR(rightGhost);
        side=BND_RIGHT;
        err |= clSetKernelArg(kernel,3,sizeof(cl_mem),&hRightNeighbour);
        err |= clSetKernelArg(kernel,4,sizeof(cl_mem),&huRightNeighbour);
        err |= clSetKernelArg(kernel,5,sizeof(cl_mem),&hvRightNeighbour);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

        err = clEnqueueNDRangeKernel(queue,kernel,1,NULL,&global,&workGroupSize,1,&leftEvent,&rightEvent);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);


        global = nX;
        adjustGlobalDim(global,workGroupSize);
        const auto hTopNeighbour = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_H_PTR(topGhost);
        const auto huTopNeighbour = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_HU_PTR(topGhost);
        const auto hvTopNeighbour = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_HV_PTR(topGhost);
        side=BND_TOP;
        err |= clSetKernelArg(kernel,3,sizeof(cl_mem),&hTopNeighbour);
        err |= clSetKernelArg(kernel,4,sizeof(cl_mem),&huTopNeighbour);
        err |= clSetKernelArg(kernel,5,sizeof(cl_mem),&hvTopNeighbour);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

        err = clEnqueueNDRangeKernel(queue,kernel,1,NULL,&global,&workGroupSize,1,&rightEvent,&botEvent);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

        const auto hBottomNeighbour = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_H_PTR(bottomGhost);
        const auto huBottomNeighbour = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_HU_PTR(bottomGhost);
        const auto hvBottomNeighbour = (cl_mem) STARPU_SWE_HUV_MATRIX_GET_HV_PTR(bottomGhost);
        side=BND_BOTTOM;
        err |= clSetKernelArg(kernel,3,sizeof(cl_mem),&hBottomNeighbour);
        err |= clSetKernelArg(kernel,4,sizeof(cl_mem),&huBottomNeighbour);
        err |= clSetKernelArg(kernel,5,sizeof(cl_mem),&hvBottomNeighbour);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

        err = clEnqueueNDRangeKernel(queue,kernel,1,NULL,&global,&workGroupSize,1,&botEvent,&topEvent);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

        starpu_opencl_release_kernel(kernel);
    }
}

void variableMin_opencl(void* buffers[],void*cl_args){
    const cl_mem a = (cl_mem) STARPU_VARIABLE_GET_PTR(buffers[0]);
    const cl_mem b = (cl_mem) STARPU_VARIABLE_GET_PTR(buffers[1]);

    int id, devid, err;                   /* OpenCL specific code */
    cl_kernel kernel;                     /* OpenCL specific code */
    cl_command_queue queue;               /* OpenCL specific code */
    cl_event event;                       /* OpenCL specific code */

    id = starpu_worker_get_id();
    devid = starpu_worker_get_devid(id);
    err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_programs,
                                    "variableMin_opencl_kernel", /* Name of the codelet */
                                    devid);
    if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    err |= clSetKernelArg(kernel,0,sizeof(cl_mem),&a);
    err |= clSetKernelArg(kernel,1,sizeof(cl_mem),&b);
    if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

    size_t global = 1;
    err = clEnqueueNDRangeKernel(queue,kernel,1,NULL,&global,&global,0,NULL,&event);
    if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

    starpu_opencl_release_kernel(kernel);

}

void variableSetInf_opencl(void* buffers[],void*cl_args){
    auto a = (cl_mem) STARPU_VARIABLE_GET_DEV_HANDLE(buffers[0]);

    int id, devid, err;                   /* OpenCL specific code */
    cl_kernel kernel;                     /* OpenCL specific code */
    cl_command_queue queue;               /* OpenCL specific code */
    cl_event event;                       /* OpenCL specific code */

    id = starpu_worker_get_id();
    devid = starpu_worker_get_devid(id);
    err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_programs,
                                    "variableSetInf_opencl_kernel", /* Name of the codelet */
                                    devid);
    if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    err |= clSetKernelArg(kernel,0,sizeof(a),&a);
    if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

    size_t global = 1;
    err = clEnqueueNDRangeKernel(queue,kernel,1,NULL,&global,&global,0,NULL,&event);
    if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);


    starpu_opencl_release_kernel(kernel);
}