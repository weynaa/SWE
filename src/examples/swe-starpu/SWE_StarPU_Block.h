//
// Created by michael on 09.05.20.
//

#ifndef SWE_SWE_STARPU_BLOCK_H
#define SWE_SWE_STARPU_BLOCK_H

#include "tools/help.hh"
#include "scenarios/SWE_Scenario.hh"
#include "starpu/StarPUCommon.h"
#include "starpu/SWE_HUV_Matrix.h"

#include <iostream>
#include <fstream>
#include <memory>

//using namespace std;

// forward declaration
/**
 * SWE_Block1D is a simple struct that can represent a single line or row of
 * SWE_Block unknowns (using the Float1D proxy class).
 * It is intended to unify the implementation of inflow and periodic boundary
 * conditions, as well as the ghost/copy-layer connection between several SWE_Block
 * grids.
 */

/**
 * Helper allocation Structure for a HUV-Matrix allocation, stored in a dense, row-major fashion
 */
struct SWE_StarPU_HUV_Allocation {

    size_t nX  =0;
    size_t nY = 0;
    float_type *_h = nullptr;
    float_type *_hu = nullptr;
    float_type *_hv = nullptr;
    starpu_data_handle_t _spu_huv = nullptr;

    SWE_StarPU_HUV_Allocation() = default;

    explicit SWE_StarPU_HUV_Allocation( const size_t _nX,const size_t _nY) : nX(_nX), nY(_nY) {
        starpu_malloc((void **) &_h, sizeof(float_type) * nX*nY);
        starpu_malloc((void **) &_hu, sizeof(float_type) *  nX*nY);
        starpu_malloc((void **) &_hv, sizeof(float_type) *  nX*nY);
    }

    SWE_StarPU_HUV_Allocation(const SWE_StarPU_HUV_Allocation&) =delete;
    SWE_StarPU_HUV_Allocation(SWE_StarPU_HUV_Allocation&& rval) noexcept {
        _h = rval._h;
        _hu = rval._hu;
        _hv = rval._hv;
        nX = rval.nX;
        nY = rval.nY;
        _spu_huv = rval._spu_huv;
        rval._spu_huv = nullptr;

        rval.nX = rval.nY = 0;
        rval._h = rval._hu = rval._hv = nullptr;
    }
    SWE_StarPU_HUV_Allocation & operator=(const SWE_StarPU_HUV_Allocation&) = delete;
    SWE_StarPU_HUV_Allocation & operator=(SWE_StarPU_HUV_Allocation && rval) noexcept {
        _h = rval._h;
        _hu = rval._hu;
        _hv = rval._hv;
        nX = rval.nX;
        nY = rval.nY;
        _spu_huv = rval._spu_huv;
        rval._spu_huv = nullptr;

        rval.nX = rval.nY = 0;
        rval._h = rval._hu = rval._hv = nullptr;
        return *this;
    }
    //Zeroes all buffers
    void clearData(){
        memset(_h,0,nX*nY*sizeof(float_type));
        memset(_hu,0,nX*nY*sizeof(float_type));
        memset(_hv,0,nX*nY*sizeof(float_type));
    }

    explicit operator bool() const {
        return nX == 0 || nY == 0 || _h == nullptr || _hu == nullptr || _hv == nullptr;
    }

    float_type & h(const size_t x,const size_t y) noexcept {
        return _h[y*nX+x];
    }
    float_type & hu(const size_t x,const size_t y) noexcept {
        return _hu[y*nX+x];
    }
    float_type & hv(const size_t x,const size_t y) noexcept {
        return _hv[y*nX+x];
    }

    void register_starpu() {
        unregister_starpu();
        starpu_swe_huv_matrix_register(&_spu_huv, STARPU_MAIN_RAM, _h, _hu, _hv,nX,nX, nY);
    }

    starpu_data_handle_t starpuHandle() const noexcept {
        return _spu_huv;
    }

    void unregister_starpu() {
        if (_spu_huv) {
            starpu_data_unregister(_spu_huv);
            _spu_huv = nullptr;
        }
    }

    ~SWE_StarPU_HUV_Allocation() {
        unregister_starpu();
        if(_h) {
            starpu_free(_h);
        }
        if(_hu) {
            starpu_free(_hu);
        }
        if(_hv) {
            starpu_free(_hv);
        }
    }
};

class SWE_StarPU_Block {
public:

    // object methods
    /// initialise unknowns to a specific scenario:
    void initScenario(float _offsetX, float _offsetY,
                      SWE_Scenario &i_scenario, bool i_multipleBlocks = false);

    // set unknowns
    /// set the water height according to a given function
    void setWaterHeight(float (*_h)(float, float));

    /// set the momentum/discharge according to the provided functions
    void setDischarge(float (*_u)(float, float), float (*_v)(float, float));

    /// set the bathymetry to a uniform value
    void setBathymetry(float _b);

    /// set the bathymetry according to a given function
    void setBathymetry(float (*_b)(float, float));

    // defining boundary conditions
    /// set type of boundary condition for the specified boundary
    void setBoundaryType(BoundaryEdge edge, BoundaryType boundtype,
                         SWE_StarPU_Block *inflow = nullptr);
//     void connectBoundaries(BoundaryEdge edge, SWE_Block &neighbour, BoundaryEdge neighEdge);


    /// return maximum size of the time step to ensure stability of the method
    /**
     * @return	current value of the member variable #maxTimestep
     */
    float getMaxTimestep() const noexcept { return maxTimestep; };


    // access methods to grid sizes
    /// returns #nx, i.e. the grid size in x-direction
    int getNx() const noexcept { return nx; }

    /// returns #ny, i.e. the grid size in y-direction
    int getNy() const noexcept { return ny; }

    float getDx() const noexcept {return dx;}

    float getDy() const noexcept {return dy;}

    // Konstanten:
    /// static variable that holds the gravity constant (g = 9.81 m/s^2):
    static constexpr float g = 9.81;


    inline float_type  & b(const size_t x, const size_t y) const noexcept {
        assert(y < ny+2ull && x < nx+2ull);
        return _b[y*(nx+2)+x];
    }

    // Constructor und Destructor
    SWE_StarPU_Block(int l_nx, int l_ny,
                     float l_dx, float l_dy);
    SWE_StarPU_Block(SWE_StarPU_Block && rval){
        nx = rval.nx;
        ny = rval.ny;
        dx = rval.dx;
        dy = rval.dy;
        _b = rval._b;
        rval._b = nullptr;
        huv_Block = std::move(rval.huv_Block);
        spu_b = rval.spu_b;
        rval.spu_b= nullptr;
        maxTimestep = rval.maxTimestep;
        offsetX = rval.offsetX;
        offsetY = rval.offsetY;
        for(int i = 0; i < 4;++i){
            boundary[i] = rval.boundary[i];
            boundaryData[i] = std::move(rval.boundaryData[i]);
            neighbours[i] = rval.neighbours[i];
        }
    }

    void starpu_unregister() {
        boundaryData[BND_LEFT].unregister_starpu();
        boundaryData[BND_TOP].unregister_starpu();
        boundaryData[BND_BOTTOM].unregister_starpu();
        boundaryData[BND_RIGHT].unregister_starpu();
        huv_Block.unregister_starpu();
        if (spu_b) {
            starpu_data_unregister(spu_b);
            spu_b = nullptr;
        }
    }

    virtual ~SWE_StarPU_Block() {
        starpu_unregister();
        if (_b) {
            starpu_free(_b);
        }

    }

    void register_starpu() {
        huv_Block.register_starpu();
        starpu_matrix_data_register(&spu_b, STARPU_MAIN_RAM,
                                    (uintptr_t) _b, nx + 2, nx+2, ny + 2, sizeof(decltype(*_b)));
        boundaryData[BND_LEFT].register_starpu();
        boundaryData[BND_TOP].register_starpu();
        boundaryData[BND_BOTTOM].register_starpu();
        boundaryData[BND_RIGHT].register_starpu();

    }
    const SWE_StarPU_HUV_Allocation & huvData() const noexcept {
        return huv_Block;
    }
    starpu_data_handle_t bStarpuHandle() const noexcept {
        return spu_b;
    }

    /// type of boundary conditions at LEFT, RIGHT, TOP, and BOTTOM boundary
    BoundaryType boundary[4];
    /// for CONNECT boundaries: pointer to connected neighbour block
    SWE_StarPU_HUV_Allocation boundaryData[4];
    SWE_StarPU_Block *neighbours[4];
protected:
    // Sets the bathymetry on outflow and wall boundaries
    void setBoundaryBathymetry();

    // grid size: number of cells (incl. ghost layer in x and y direction:
    int nx;    ///< size of Cartesian arrays in x-direction
    int ny;    ///< size of Cartesian arrays in y-direction
    // mesh size dx and dy:
    float dx;    ///<  mesh size of the Cartesian grid in x-direction
    float dy;    ///<  mesh size of the Cartesian grid in y-direction

    // define arrays for unknowns:
    // h (water level) and u,v (velocity in x and y direction)
    float *_b = nullptr;  ///< array that holds the bathymetry data (sea floor elevation) for each element

    SWE_StarPU_HUV_Allocation huv_Block;
    starpu_data_handle_t spu_b = nullptr;

    /// maximum time step allowed to ensure stability of the method
    /**
     * maxTimestep can be updated as part of the methods computeNumericalFluxes
     * and updateUnknowns (depending on the numerical method)
     */
    float maxTimestep;

    // offset of current block
    float offsetX;    ///< x-coordinate of the origin (left-bottom corner) of the Cartesian grid
    float offsetY;    ///< y-coordinate of the origin (left-bottom corner) of the Cartesian grid
};

#endif //SWE_SWE_STARPU_BLOCK_H
