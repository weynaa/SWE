//
// Created by michael on 12.05.20.
//

#include "SWE_HUV_Matrix.h"

static void swe_huv_matrix_register_data_handle(starpu_data_handle_t handle, unsigned homeNode, void *dataInterface) {
    auto *interface = (SWE_HUV_Matrix_interface *) dataInterface;
    for (unsigned node = 0; node < STARPU_MAXNODES; node++) {
        SWE_HUV_Matrix_interface *local_interface = (SWE_HUV_Matrix_interface *)
                starpu_data_get_interface_on_node(handle, node);
        local_interface->nX = interface->nX;
        local_interface->nY = interface->nY;
        local_interface->ld = interface->ld;
        if (node == homeNode) {
            local_interface->h = interface->h;
            local_interface->hu = interface->hu;
            local_interface->hv = interface->hv;
        } else {
            local_interface->h = nullptr;
            local_interface->hu = nullptr;
            local_interface->hv = nullptr;
        }
    }
}

static starpu_ssize_t swe_huv_matrix_allocate_data_on_node(void *dataInterface, unsigned node) {
    auto interface = (SWE_HUV_Matrix_interface *) dataInterface;
    const auto requestedSpace = sizeof(float_type) * interface->nX * interface->nY;

    float_type *h = nullptr;
    float_type *hu = nullptr;
    float_type *hv = nullptr;

    h = (float_type *) starpu_malloc_on_node(node, requestedSpace);
    if (!h) {
        return -ENOMEM;
    }
    hu = (float_type *) starpu_malloc_on_node(node, requestedSpace);
    if (!hu) {
        starpu_free_on_node(node, (uintptr_t) h, requestedSpace);
        return -ENOMEM;
    }
    hv = (float_type *) starpu_malloc_on_node(node, requestedSpace);
    if (!hv) {
        starpu_free_on_node(node, (uintptr_t) h, requestedSpace);
        starpu_free_on_node(node, (uintptr_t) hu, requestedSpace);
        return -ENOMEM;
    }
    interface->h = h;
    interface->hu = hu;
    interface->hv = hv;

    return requestedSpace * 3;
}

static void swe_huv_matrix_free_data_on_node(void *data_interface, unsigned node) {
    auto *interface = (SWE_HUV_Matrix_interface *) data_interface;
    starpu_ssize_t requested_memory = interface->nX * interface->nY * sizeof(interface->h[0]);

    starpu_free_on_node(node, (uintptr_t) interface->h, requested_memory);
    starpu_free_on_node(node, (uintptr_t) interface->hu, requested_memory);
    starpu_free_on_node(node, (uintptr_t) interface->hv, requested_memory);
}

static int copy_any_to_any(void *src_interface, unsigned src_node,
                           void *dst_interface, unsigned dst_node,
                           void *async_data) {
    auto src = (SWE_HUV_Matrix_interface *) src_interface;
    auto dst = (SWE_HUV_Matrix_interface *) dst_interface;
    int ret = 0;

    if (starpu_interface_copy((uintptr_t) src->h, 0, src_node,
                              (uintptr_t) dst->h, 0, dst_node,
                              src->nX * src->nY * sizeof(src->h[0]),
                              async_data)) {
        ret = -EAGAIN;
    }
    if (starpu_interface_copy((uintptr_t) src->hu, 0, src_node,
                              (uintptr_t) dst->hu, 0, dst_node,
                              src->nX * src->nY * sizeof(src->hu[0]),
                              async_data)) {
        ret = -EAGAIN;
    }
    if (starpu_interface_copy((uintptr_t) src->hv, 0, src_node,
                              (uintptr_t) dst->hv, 0, dst_node,
                              src->nX * src->nY * sizeof(src->hv[0]),
                              async_data)) {
        ret = -EAGAIN;
    }
    return ret;
}

static const struct starpu_data_copy_methods swe_huv_matrix_copy_methods = []() {
    starpu_data_copy_methods methods = {};
    methods.any_to_any = copy_any_to_any;
    return methods;
}();

static size_t swe_huv_matrix_get_size(starpu_data_handle_t handle) {
    auto interface = (SWE_HUV_Matrix_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
    return 3 * interface->nY * interface->nX * sizeof(interface->h[0]);
}

static uint32_t swe_huv_matrix_footprint(starpu_data_handle_t handle) {
    auto interface = (SWE_HUV_Matrix_interface *)
            starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
    return starpu_hash_crc32c_be(interface->nX * interface->nY, 0);
}

static int swe_huv_matrix_pointer_is_inside(void *dataInterface, unsigned node, void *ptr) {
    auto interface = (const SWE_HUV_Matrix_interface *) dataInterface;

    return ((char *) ptr >= (char *) interface->h &&
            (char *) ptr < (char *) (interface->h + interface->nX * interface->nY))
           || ((char *) ptr >= (char *) interface->hu &&
               (char *) ptr < (char *) (interface->hu + interface->nX * interface->nY))
           || ((char *) ptr >= (char *) interface->hv &&
               (char *) ptr < (char *) (interface->hv + interface->nX * interface->nY))
           || ((char *) ptr >= (char *) &interface &&
               (char *) ptr < (char *) (&interface) + sizeof(*interface));
}

void starpu_swe_huv_matrix_register(starpu_data_handle_t *outHandle, unsigned homeNode,
                                    float *h, float *hu, float *hv, const size_t nX, const size_t ld, const size_t nY) {
    //This is C++11 so we have no named intialization, but we can hack something with lambdas
    static starpu_data_interface_ops swe_huv_matrix_interface_ops = []() {
        starpu_data_interface_ops interface = {};
        interface.register_data_handle = swe_huv_matrix_register_data_handle;
        interface.allocate_data_on_node = swe_huv_matrix_allocate_data_on_node;
        interface.free_data_on_node = swe_huv_matrix_free_data_on_node;
        interface.copy_methods = &swe_huv_matrix_copy_methods;
        interface.get_size = swe_huv_matrix_get_size;
        interface.footprint = swe_huv_matrix_footprint;
        interface.interfaceid = static_cast<starpu_data_interface_id>(starpu_data_interface_get_next_id());
        interface.interface_size = sizeof(SWE_HUV_Matrix_interface);
        interface.to_pointer = nullptr;
        interface.pointer_is_inside = swe_huv_matrix_pointer_is_inside;
        return interface;
    }();
    SWE_HUV_Matrix_interface matrix = {};
    matrix.h = h;
    matrix.hu = hu;
    matrix.hv = hv;
    matrix.nX = nX;
    matrix.nY = nY;
    matrix.ld = ld;

    starpu_data_register(outHandle, homeNode, &matrix, &swe_huv_matrix_interface_ops);

}
