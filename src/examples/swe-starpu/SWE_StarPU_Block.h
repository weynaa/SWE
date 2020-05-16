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

        rval.nX = rval.nY = 0;
        rval._h = rval._hu = rval._hv = nullptr;
        return *this;
    }

    explicit operator bool() const {
        return nX == 0 || nY == 0 || _h == nullptr || _hu == nullptr || _hv == nullptr;
    }

    float & h(const size_t x,const size_t y) const noexcept {
        return _h[y*nX+x];
    }
    float & hu(const size_t x,const size_t y) const noexcept {
        return _hu[y*nX+x];
    }
    float & hv(const size_t x,const size_t y) const noexcept {
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

/**
 * SWE_Block is the main data structure to compute our shallow water model
 * on a single Cartesian grid block:
 * SWE_Block is an abstract class (and interface) that should be extended
 * by respective implementation classes.
 *
 * <h3>Cartesian Grid for Discretization:</h3>
 *
 * SWE_Blocks uses a regular Cartesian grid of size #nx by #ny, where each
 * grid cell carries three unknowns:
 * - the water level #h
 * - the momentum components #hu and #hv (in x- and y- direction, resp.)
 * - the bathymetry #b
 *
 * Each of the components is stored as a 2D array, implemented as a Float2D object,
 * and are defined on grid indices [0,..,#nx+1]*[0,..,#ny+1].
 * The computational domain is indexed with [1,..,#nx]*[1,..,#ny].
 *
 * The mesh sizes of the grid in x- and y-direction are stored in static variables
 * #dx and #dy. The position of the Cartesian grid in space is stored via the
 * coordinates of the left-bottom corner of the grid, in the variables
 * #offsetX and #offsetY.
 *
 * <h3>Ghost layers:</h3>
 *
 * To implement the behaviour of the fluid at boundaries and for using
 * multiple block in serial and parallel settings, SWE_Block adds an
 * additional layer of so-called ghost cells to the Cartesian grid,
 * as illustrated in the following figure.
 * Cells in the ghost layer have indices 0 or #nx+1 / #ny+1.
 *
 * \image html ghost_cells.gif
 *
 * <h3>Memory Model:</h3>
 *
 * The variables #h, #hu, #hv for water height and momentum will typically be
 * updated by classes derived from SWE_Block. However, it is not assumed that
 * such and updated will be performed in every time step.
 * Instead, subclasses are welcome to update #h, #hu, and #hv in a lazy fashion,
 * and keep data in faster memory (incl. local memory of acceleration hardware,
 * such as GPGPUs), instead.
 *
 * It is assumed that the bathymetry data #b is not changed during the algorithm
 * (up to the exceptions mentioned in the following).
 *
 * To force a synchronization of the respective data structures, the following
 * methods are provided as part of SWE_Block:
 * - synchAfterWrite() to synchronize #h, #hu, #hv, and #b after an external update
 *   (reading a file, e.g.);
 * - synchWaterHeightAfterWrite(), synchDischargeAfterWrite(), synchBathymetryAfterWrite():
 *   to synchronize only #h or momentum (#hu and #hv) or bathymetry #b;
 * - synchGhostLayerAfterWrite() to synchronize only the ghost layers
 * - synchBeforeRead() to synchronize #h, #hu, #hv, and #b before an output of the
 *   variables (writing a visualization file, e.g.)
 * - synchWaterHeightBeforeRead(), synchDischargeBeforeRead(), synchBathymetryBeforeRead():
 *   as synchBeforeRead(), but only for the specified variables
 * - synchCopyLayerBeforeRead(): synchronizes the copy layer only (i.e., a layer that
 *   is to be replicated in a neighbouring SWE_Block.
 *
 * <h3>Derived Classes</h3>
 *
 * As SWE_Block just provides an abstract base class together with the most
 * important data structures, the implementation of concrete models is the
 * job of respective derived classes (see the class diagram at the top of this
 * page). Similar, parallel implementations that are based on a specific
 * parallel programming model (such as OpenMP) or parallel architecture
 * (such as GPU/CUDA) should form subclasses of their own.
 * Please refer to the documentation of these classes for more details on the
 * model and on the parallelisation approach.
 */
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
    float *_b;  ///< array that holds the bathymetry data (sea floor elevation) for each element

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
