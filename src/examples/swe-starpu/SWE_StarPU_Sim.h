#ifndef SWE_SWE_STARPU_SIM_H
#define SWE_SWE_STARPU_SIM_H

#include <vector>
#include <atomic>
#include "SWE_StarPU_Block.h"
#include <writer/StarPUBlockWriter.h>
#include "codelets.h"
#include <tools/help.hh>

constexpr int nLayers = 2;

struct SWE_StarPU_Sim {
    const size_t nX;
    const size_t nY;
    const size_t nBlocksX;
    const size_t nBlocksY;
    const float t_begin;
    const float t_end;
private:
    //Layer, X, Y 3D array. Layer is used to have inter-timestep parallelism
    std::vector<std::vector<SWE_StarPU_Block>> blocks;
    std::vector<std::vector<starpu_data_handle_t>> scratchData;
    std::vector<std::vector<io::StarPUBlockWriter>> writers;


    float timestamp = 0;
    starpu_data_handle_t spu_timestamp;
    float nextTimestampToWrite = 0;
    starpu_data_handle_t spu_nextTimestampToWrite;
    std::atomic<int> iterationNumber = {0};

    starpu_data_handle_t spu_maxTimestep;

    std::vector<float> l_checkPoints;

    constexpr static size_t ITERATION_STEPS_TO_SCHEDULE = 10;

public:
    explicit SWE_StarPU_Sim(const size_t _nX, const size_t _nY,
                            const size_t _nBlocksX, const size_t _nBlocksY,
                            const float dX, const float dY,
                            std::string &filename,
                            SWE_Scenario &scenario,
                            const int numberOfCheckpoints = 20);

    ~SWE_StarPU_Sim();

    void launchTaskGraph();

    void writeTimeStep();

    void updateGhostLayers();

    void computeNumericalFluxes() ;

    void updateUnkowns();

    void runTimestep();

};


#endif //SWE_SWE_STARPU_SIM_H
