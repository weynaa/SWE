#include "codelets_cl.h"
#include <starpu/StarPUCommon.h>
#include "SWE_StarPU_Block.h"
#include <iostream>
#include <cmath>

extern struct starpu_opencl_program opencl_programs;

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