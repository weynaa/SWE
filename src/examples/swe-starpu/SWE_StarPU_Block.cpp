#include "SWE_StarPU_Block.h"
#include "tools/help.hh"

#include <cmath>
#include <iostream>
#include <cassert>
#include <limits>
#include <type_traits>
#include <algorithm>


/**
 * Constructor: allocate variables for simulation
 *
 * unknowns h (water height), hu,hv (discharge in x- and y-direction),
 * and b (bathymetry) are defined on grid indices [0,..,nx+1]*[0,..,ny+1]
 * -> computational domain is [1,..,nx]*[1,..,ny]
 * -> plus ghost cell layer
 *
 * The constructor is protected: no instances of SWE_Block can be
 * generated.
 *
 */
SWE_StarPU_Block::SWE_StarPU_Block(int l_nx, int l_ny,
                     float l_dx, float l_dy)
        : nx(l_nx), ny(l_ny),
          dx(l_dx), dy(l_dy),

        // This three are only set here, so eclipse does not complain
          maxTimestep(0), offsetX(0), offsetY(0)
{
    starpu_malloc((void**)&h,sizeof(float_type)*nx*ny);
    starpu_malloc((void**)&hu,sizeof(float_type)*(nx+2)*(ny+2));
    starpu_malloc((void**)&hv,sizeof(float_type)*(nx+2)*(ny+2));
    starpu_malloc((void**)&b,sizeof(float_type)*(nx+2)*(ny+2));
    // set WALL as default boundary condition
    for (int i=0; i<4; i++) {
        boundary[i] = PASSIVE;
        neighbours[i] = NULL;
    };

}


//==================================================================
// methods for external read/write to main variables h, hu, hv, and b
// Note: temporary and non-local variables depending on the main
// variables are synchronised before/after their update or read
//==================================================================

/**
 * Initializes the unknowns and bathymetry in all grid cells according to the given SWE_Scenario.
 *
 * In the case of multiple SWE_Blocks at this point, it is not clear how the boundary conditions
 * should be set. This is because an isolated SWE_Block doesn't have any in information about the grid.
 * Therefore the calling routine, which has the information about multiple blocks, has to take care about setting
 * the right boundary conditions.
 *
 * @param i_scenario scenario, which is used during the setup.
 * @param i_multipleBlocks are the multiple SWE_blocks?
 */
void SWE_StarPU_Block::initScenario( float _offsetX, float _offsetY,
                              SWE_Scenario &i_scenario,
                              const bool i_multipleBlocks ) {
    offsetX = _offsetX;
    offsetY = _offsetY;

    // initialize water height and discharge
    for(int i=1; i<=nx; i++)
        for(int j=1; j<=ny; j++) {
            float x = offsetX + (i-0.5f)*dx;
            float y = offsetY + (j-0.5f)*dy;
            h[i*rowStride()+j] =  i_scenario.getWaterHeight(x,y);
            hu[i*rowStride()+j] = i_scenario.getVeloc_u(x,y) * h[i*rowStride()+j];
            hv[i*rowStride()+j] = i_scenario.getVeloc_v(x,y) * h[i*rowStride()+j];
        };

    // initialize bathymetry
    for(int i=0; i<=nx+1; i++) {
        for(int j=0; j<=ny+1; j++) {
            b[i*rowStride()+j] = i_scenario.getBathymetry( offsetX + (i-0.5f)*dx,
                                                offsetY + (j-0.5f)*dy );
        }
    }

    // in the case of multiple blocks the calling routine takes care about proper boundary conditions.
    if( !i_multipleBlocks) {
        // obtain boundary conditions for all four edges from scenario
        setBoundaryType(BND_LEFT, i_scenario.getBoundaryType(BND_LEFT));
        setBoundaryType(BND_RIGHT, i_scenario.getBoundaryType(BND_RIGHT));
        setBoundaryType(BND_BOTTOM, i_scenario.getBoundaryType(BND_BOTTOM));
        setBoundaryType(BND_TOP, i_scenario.getBoundaryType(BND_TOP));
    }

}

/**
 * set water height h in all interior grid cells (i.e. except ghost layer)
 * to values specified by parameter function _h
 */
void SWE_StarPU_Block::setWaterHeight(float (*_h)(float, float)) {

    for(int i=1; i<=nx; i++)
        for(int j=1; j<=ny; j++) {
            h[i*rowStride()+j] =  _h(offsetX + (i-0.5f)*dx, offsetY + (j-0.5f)*dy);
        };
}

/**
 * set discharge in all interior grid cells (i.e. except ghost layer)
 * to values specified by parameter functions
 * Note: unknowns hu and hv represent momentum, while parameters u and v are velocities!
 */
void SWE_StarPU_Block::setDischarge(float (*_u)(float, float), float (*_v)(float, float)) {

    for(int i=1; i<=nx; i++)
        for(int j=1; j<=ny; j++) {
            float x = offsetX + (i-0.5f)*dx;
            float y = offsetY + (j-0.5f)*dy;
            hu[i*rowStride()+j] = _u(x,y) * h[i*rowStride()+j];
            hv[i*rowStride()+j] = _v(x,y) * h[i*rowStride()+j];
        };

}

/**
 * set Bathymetry b in all grid cells (incl. ghost/boundary layers)
 * to a uniform value
 * bathymetry source terms are re-computed
 */
void SWE_StarPU_Block::setBathymetry(float _b) {

    for(int i=0; i<=nx+1; i++)
        for(int j=0; j<=ny+1; j++)
            b[i*rowStride()+j] = _b;

}

/**
 * set Bathymetry b in all grid cells (incl. ghost/boundary layers)
 * using the specified bathymetry function;
 * bathymetry source terms are re-computed
 */
void SWE_StarPU_Block::setBathymetry(float (*_b)(float, float)) {

    for(int i=0; i<=nx+1; i++)
        for(int j=0; j<=ny+1; j++)
            b[i*rowStride()+j] = _b(offsetX + (i-0.5f)*dx, offsetY + (j-0.5f)*dy);

}

//==================================================================
// methods for simulation
//==================================================================

/**
 * Set the boundary type for specific block boundary.
 *
 * @param i_edge location of the edge relative to the SWE_block.
 * @param i_boundaryType type of the boundary condition.
 * @param i_inflow pointer to an SWE_Block1D, which specifies the inflow (should be NULL for WALL or OUTFLOW boundary)
 */
void SWE_StarPU_Block::setBoundaryType( const BoundaryEdge i_edge,
                                 const BoundaryType i_boundaryType,
                                 SWE_StarPU_Block* i_inflow) {
    boundary[i_edge] = i_boundaryType;
    neighbours[i_edge] = i_inflow;

    if (i_boundaryType == OUTFLOW || i_boundaryType == WALL)
        // One of the boundary was changed to OUTFLOW or WALL
        // -> Update the bathymetry for this boundary
        setBoundaryBathymetry();
}

/**
 * Sets the bathymetry on OUTFLOW or WALL boundaries.
 * Should be called very time a boundary is changed to a OUTFLOW or
 * WALL boundary <b>or</b> the bathymetry changes.
 */
void SWE_StarPU_Block::setBoundaryBathymetry()
{
    // set bathymetry values in the ghost layer, if necessary
    if( boundary[BND_LEFT] == OUTFLOW || boundary[BND_LEFT] == WALL ) {
        memcpy(&b[0], &b[1*rowStride()], sizeof(float)*(ny+2));
    }
    if( boundary[BND_RIGHT] == OUTFLOW || boundary[BND_RIGHT] == WALL ) {
        memcpy(&b[(nx+1)*rowStride()], &b[nx*rowStride()], sizeof(float)*(ny+2));
    }
    if( boundary[BND_BOTTOM] == OUTFLOW || boundary[BND_BOTTOM] == WALL ) {
        for(int i=0; i<=nx+1; i++) {
            b[i*rowStride()+0] = b[i*rowStride()+1];
        }
    }
    if( boundary[BND_TOP] == OUTFLOW || boundary[BND_TOP] == WALL ) {
        for(int i=0; i<=nx+1; i++) {
            b[i*rowStride()+ny+1] = b[i+rowStride()+ny];
        }
    }


    // set corner values
    b[0+rowStride()+0]       = b[1+rowStride()+1];
    b[0*rowStride()+ny+1]    = b[1*rowStride()+ny];
    b[(nx+1)*rowStride()+0]    = b[nx*rowStride()+1];
    b[(nx+1)*rowStride()+ny+1] = b[nx*rowStride()+ny];
}