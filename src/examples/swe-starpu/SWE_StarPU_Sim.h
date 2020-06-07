#ifndef SWE_SWE_STARPU_SIM_H
#define SWE_SWE_STARPU_SIM_H

#include <vector>
#include <atomic>
#include "SWE_StarPU_Block.h"
#include <writer/StarPUBlockWriter.h>
#include "codelets.h"
#include <tools/help.hh>

constexpr int nLayers = 3;

struct SWE_StarPU_Sim {
    const size_t nX;
    const size_t nY;
    const size_t nBlocksX;
    const size_t nBlocksY;
    const float t_begin;
    const float t_end;


private:
    //Layer, X, Y 3D array. Layer is used to have some inter-timestep parallelism
    std::array<std::vector<std::vector<SWE_StarPU_Block>>,nLayers> blocks;
    std::array<std::vector<std::vector<starpu_data_handle_t>>,nLayers> scratchData;
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
                            const int numberOfCheckpoints = 20)
            : nX(_nX), nY(_nY), nBlocksX(_nBlocksX), nBlocksY(_nBlocksY),
              t_begin(0), t_end(scenario.endSimulation()) {

        const auto boundsWidth = (scenario.getBoundaryPos(BND_RIGHT) -
                                  scenario.getBoundaryPos(BND_LEFT));
        const auto boundsHeight = (scenario.getBoundaryPos(BND_TOP) -
                                   scenario.getBoundaryPos(BND_BOTTOM));

        for (int cp = 1; cp <= numberOfCheckpoints; cp++) {
            l_checkPoints.push_back((float) cp * (scenario.endSimulation() / numberOfCheckpoints));
        }
        nextTimestampToWrite = l_checkPoints.front();
        starpu_variable_data_register(&spu_timestamp, STARPU_MAIN_RAM,
                                      (uintptr_t) &timestamp, sizeof(timestamp));
        starpu_variable_data_register(&spu_nextTimestampToWrite, STARPU_MAIN_RAM,
                                      (uintptr_t) &nextTimestampToWrite, sizeof(nextTimestampToWrite));
        starpu_variable_data_register(&spu_maxTimestep, -1,
                                      0, sizeof(float));
        starpu_data_set_reduction_methods(spu_maxTimestep, &SWECodelets::variableMin,
                                          &SWECodelets::variableSetInf);
        for(size_t layer = 0; layer < blocks.size();++layer) {
            auto & blk = blocks[layer];
            auto & scratch = scratchData[layer];
            for (size_t x = 0; x < nBlocksX; ++x) {
                blk.emplace_back();
                auto &row = blk.back();
                if(layer == 0) {
                    writers.emplace_back();
                }
                auto &writerRow = writers.back();
                scratch.emplace_back();
                auto &scratchRow = scratch.back();
                for (size_t y = 0; y < nBlocksY; ++y) {
                    const auto blockNx = x == nBlocksX - 1 ? nX - x * (nX / nBlocksX) : nX / nBlocksX;
                    const auto blockNy = y == nBlocksY - 1 ? nY - y * (nY / nBlocksY) : nY / nBlocksY;
                    row.emplace_back(
                            (int) blockNx,
                            (int) blockNy,
                            (float) dX,
                            (float) dY
                    );

                    const auto offsetX = (boundsWidth / nBlocksX) * x;
                    const auto offsetY = (boundsHeight / nBlocksY) * y;
                    row.back().initScenario(offsetX, offsetY, scenario,
                                            true);
                    row.back().register_starpu();
                    if(layer == 0) {
                        writerRow.emplace_back(
                                generateBaseFileName(filename, (int) x, (int) y),
                                blockNx,
                                blockNy,
                                dX, dY,
                                offsetX,
                                offsetY,
                                0
                        );
                    }
                    scratchRow.push_back(nullptr);
                    starpu_swe_huv_matrix_register(&scratchRow.back(), -1,
                                                   0, 0, 0, blockNx, blockNx, blockNy);
                }
            }
            for (size_t x = 0; x < nBlocksX; ++x) {
                for (size_t y = 0; y < nBlocksY; ++y) {
                    if (x != 0) {
                        blk[x][y].neighbours[BND_LEFT] = &blk[x - 1][y];
                        blk[x][y].boundary[BND_LEFT] = BoundaryType::CONNECT;
                    } else {
                        blk[x][y].setBoundaryType(BND_LEFT, scenario.getBoundaryType(BND_LEFT));
                    }
                    if (x != nBlocksX - 1) {
                        blk[x][y].neighbours[BND_RIGHT] = &blk[x + 1][y];
                        blk[x][y].boundary[BND_RIGHT] = BoundaryType::CONNECT;
                    } else {
                        blk[x][y].setBoundaryType(BND_RIGHT, scenario.getBoundaryType(BND_RIGHT));
                    }
                    if (y != 0) {
                        blk[x][y].neighbours[BND_TOP] = &blk[x][y - 1];
                        blk[x][y].boundary[BND_TOP] = BoundaryType::CONNECT;
                    } else {
                        blk[x][y].setBoundaryType(BND_TOP, scenario.getBoundaryType(BND_TOP));
                    }
                    if (y != nBlocksY - 1) {
                        blk[x][y].neighbours[BND_BOTTOM] = &blk[x][y + 1];
                        blk[x][y].boundary[BND_BOTTOM] = BoundaryType::CONNECT;
                    } else {
                        blk[x][y].setBoundaryType(BND_BOTTOM, scenario.getBoundaryType(BND_BOTTOM));
                    }
                }
            }
        }

    }

    ~SWE_StarPU_Sim() {
        starpu_data_unregister(spu_timestamp);
        starpu_data_unregister(spu_nextTimestampToWrite);
        starpu_data_unregister(spu_maxTimestep);
        for(size_t layer = 0; layer < nLayers;++layer) {
            for (size_t x = 0; x < scratchData.size(); ++x) {
                for (size_t y = 0; y < scratchData[x].size(); ++y) {
                    starpu_data_unregister(scratchData[layer][x][y]);
                    blocks[layer][x][y].starpu_unregister();
                }
            }
            scratchData[layer].clear();
        }
    }

    void launchTaskGraph() {
        writeTimeStep();
        //Kickoff layer tasks
        for(int i = 0; i < blocks.size();++i) {
            runTimestep();
        }
    }

    void writeTimeStep() {
        const auto layer = iterationNumber%nLayers;
        for (size_t x = 0; x < writers.size(); ++x) {
            for (size_t y = 0; y < writers[x].size(); ++y) {
                const auto pWriters = &writers[x][y];
                starpu_task_insert(&SWECodelets::resultWriter,
                                   STARPU_VALUE, &pWriters, sizeof(pWriters),
                                   STARPU_R, blocks[layer][x][y].huvData().starpuHandle(),
                                   STARPU_R, blocks[layer][x][y].bStarpuHandle(),
                                   STARPU_R, spu_timestamp,
                                   0);
            }
        }
    }

    void updateGhostLayers() {
        const auto layer = iterationNumber%nLayers;
        for (size_t x = 0; x < blocks.size(); ++x) {
            for (size_t y = 0; y < blocks[x].size(); ++y) {
                auto side = BND_LEFT;
                auto blockptr = &blocks[x][y];

                starpu_task_insert(&SWECodelets::updateGhostLayers,
                                   STARPU_VALUE, &side, sizeof(side),
                                   STARPU_VALUE, &blockptr, sizeof(blockptr),
                                   STARPU_W, blocks[layer][x][y].boundaryData[side].starpuHandle(),
                                   STARPU_R, blocks[layer][x][y].huvData().starpuHandle(),
                                   STARPU_R, blocks[layer][x][y].neighbours[side]?blocks[layer][x][y].neighbours[side]->huvData().starpuHandle():blocks[layer][x][y].huvData().starpuHandle(),
                                   0);
                side = BND_RIGHT;
                starpu_task_insert(&SWECodelets::updateGhostLayers,
                                   STARPU_VALUE, &side, sizeof(side),
                                   STARPU_VALUE, &blockptr, sizeof(blockptr),
                                   STARPU_W, blocks[layer][x][y].boundaryData[side].starpuHandle(),
                                   STARPU_R, blocks[layer][x][y].huvData().starpuHandle(),
                                   STARPU_R, blocks[layer][x][y].neighbours[side]?blocks[layer][x][y].neighbours[side]->huvData().starpuHandle():blocks[layer][x][y].huvData().starpuHandle(),
                                   0);
                side = BND_BOTTOM;
                starpu_task_insert(&SWECodelets::updateGhostLayers,
                                   STARPU_VALUE, &side, sizeof(side),
                                   STARPU_VALUE, &blockptr, sizeof(blockptr),
                                   STARPU_W, blocks[layer][x][y].boundaryData[side].starpuHandle(),
                                   STARPU_R, blocks[layer][x][y].huvData().starpuHandle(),
                                   STARPU_R, blocks[layer][x][y].neighbours[side]?blocks[layer][x][y].neighbours[side]->huvData().starpuHandle():blocks[layer][x][y].huvData().starpuHandle(),
                                   0);
                side = BND_TOP;
                starpu_task_insert(&SWECodelets::updateGhostLayers,
                                   STARPU_VALUE, &side, sizeof(side),
                                   STARPU_VALUE, &blockptr, sizeof(blockptr),
                                   STARPU_W, blocks[layer][x][y].boundaryData[side].starpuHandle(),
                                   STARPU_R, blocks[layer][x][y].huvData().starpuHandle(),
                                   STARPU_R, blocks[layer][x][y].neighbours[side]?blocks[layer][x][y].neighbours[side]->huvData().starpuHandle():blocks[layer][x][y].huvData().starpuHandle(),
                                   0);
            }
        }
    }

    void computeNumericalFluxes() {
        const auto layer = iterationNumber%nLayers;
        for (size_t x = 0; x < blocks.size(); ++x) {
            for (size_t y = 0; y < blocks[x].size(); ++y) {
                const auto blockptr = &blocks[layer][x][y];
                starpu_task_insert(&SWECodelets::computeNumericalFluxes,
                                   STARPU_VALUE, &blockptr, sizeof(blockptr),
                                   STARPU_R, blocks[layer][x][y].huvData().starpuHandle(),
                                   STARPU_R, blocks[layer][x][y].boundaryData[BND_LEFT].starpuHandle(),
                                   STARPU_R, blocks[layer][x][y].boundaryData[BND_RIGHT].starpuHandle(),
                                   STARPU_R, blocks[layer][x][y].boundaryData[BND_BOTTOM].starpuHandle(),
                                   STARPU_R, blocks[layer][x][y].boundaryData[BND_TOP].starpuHandle(),
                                   STARPU_R, blocks[layer][x][y].bStarpuHandle(),
                                   STARPU_W, scratchData[layer][x][y],
                                   STARPU_REDUX, spu_maxTimestep,
                                   0);
            }
        }
    }

    void updateUnkowns() {
        const auto layer = iterationNumber%nLayers;
        for (size_t x = 0; x < blocks.size(); ++x) {
            for (size_t y = 0; y < blocks[x].size(); ++y) {
                const auto blockptr = &blocks[layer][x][y];
                starpu_task_insert(&SWECodelets::updateUnknowns,
                                   STARPU_VALUE, &blockptr, sizeof(blockptr),
                                   STARPU_RW, blocks[layer][x][y].huvData().starpuHandle(),
                                   STARPU_R, scratchData[layer][x][y],
                                   STARPU_R, spu_maxTimestep,
                                   0);
            }
        }
    }

    void runTimestep() {
        starpu_iteration_push(iterationNumber);
        updateGhostLayers();
        computeNumericalFluxes();
        updateUnkowns();

        const auto pSim = this;
        const auto pCheckpoints = &l_checkPoints;
        starpu_task_insert(&SWECodelets::incrementTime,
                           STARPU_VALUE, &pSim, sizeof(pSim),
                           STARPU_VALUE, &pCheckpoints, sizeof(pCheckpoints),
                           STARPU_RW, spu_timestamp,
                           STARPU_R, spu_maxTimestep,
                           STARPU_RW, spu_nextTimestampToWrite,
                           0);
        starpu_iteration_pop();
        iterationNumber++;
    }

};


#endif //SWE_SWE_STARPU_SIM_H
