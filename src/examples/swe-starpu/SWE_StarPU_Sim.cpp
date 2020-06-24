#include "SWE_StarPU_Sim.h"

SWE_StarPU_Sim::SWE_StarPU_Sim(const size_t _nX, const size_t _nY, const size_t _nBlocksX, const size_t _nBlocksY, const float dX, const float dY, std::string& filename, SWE_Scenario& scenario, const int numberOfCheckpoints /*= 20*/) : nX(_nX), nY(_nY), nBlocksX(_nBlocksX), nBlocksY(_nBlocksY),
t_begin(0), t_end(scenario.endSimulation())
{
	const auto boundsWidth = (scenario.getBoundaryPos(BND_RIGHT) -
		scenario.getBoundaryPos(BND_LEFT));
	const auto boundsHeight = (scenario.getBoundaryPos(BND_TOP) -
		scenario.getBoundaryPos(BND_BOTTOM));

	for (int cp = 1; cp <= numberOfCheckpoints; cp++) {
		l_checkPoints.push_back((float)cp * (scenario.endSimulation() / numberOfCheckpoints));
	}
	nextTimestampToWrite = l_checkPoints.front();
	starpu_variable_data_register(&spu_timestamp, STARPU_MAIN_RAM,
		(uintptr_t)&timestamp, sizeof(timestamp));
	starpu_variable_data_register(&spu_nextTimestampToWrite, STARPU_MAIN_RAM,
		(uintptr_t)&nextTimestampToWrite, sizeof(nextTimestampToWrite));
	starpu_variable_data_register(&spu_maxTimestep, -1,
		0, sizeof(float));
	starpu_data_set_reduction_methods(spu_maxTimestep, &SWECodelets::variableMin,
		&SWECodelets::variableSetInf);
	for (size_t x = 0; x < nBlocksX; ++x) {
		blocks.emplace_back();
		auto& row = blocks.back();
		writers.emplace_back();
		auto& writerRow = writers.back();
		scratchData.emplace_back();
		auto& scratchRow = scratchData.back();
		for (size_t y = 0; y < nBlocksY; ++y) {
			const auto blockNx = x == nBlocksX - 1 ? nX - x * (nX / nBlocksX) : nX / nBlocksX;
			const auto blockNy = y == nBlocksY - 1 ? nY - y * (nY / nBlocksY) : nY / nBlocksY;
			row.emplace_back(
				(int)blockNx,
				(int)blockNy,
				(float)dX,
				(float)dY
			);

			const auto offsetX = (boundsWidth / nBlocksX) * x;
			const auto offsetY = (boundsHeight / nBlocksY) * y;
			row.back().initScenario(offsetX, offsetY, scenario,
				true);
			row.back().register_starpu();
			writerRow.emplace_back(
				generateBaseFileName(filename, (int)x, (int)y),
				blockNx,
				blockNy,
				dX, dY,
				offsetX,
				offsetY,
				0
			);
			scratchRow.push_back(nullptr);
			starpu_swe_huv_matrix_register(&scratchRow.back(), -1,
				0, 0, 0, blockNx, blockNx, blockNy);
		}
	}
	for (size_t x = 0; x < nBlocksX; ++x) {
		for (size_t y = 0; y < nBlocksY; ++y) {
			if (x != 0) {
				blocks[x][y].neighbours[BND_LEFT] = &blocks[x - 1][y];
				blocks[x][y].boundary[BND_LEFT] = BoundaryType::CONNECT;
			}
			else {
				blocks[x][y].setBoundaryType(BND_LEFT, scenario.getBoundaryType(BND_LEFT));
			}
			if (x != nBlocksX - 1) {
				blocks[x][y].neighbours[BND_RIGHT] = &blocks[x + 1][y];
				blocks[x][y].boundary[BND_RIGHT] = BoundaryType::CONNECT;
			}
			else {
				blocks[x][y].setBoundaryType(BND_RIGHT, scenario.getBoundaryType(BND_RIGHT));
			}
			if (y != 0) {
				blocks[x][y].neighbours[BND_TOP] = &blocks[x][y - 1];
				blocks[x][y].boundary[BND_TOP] = BoundaryType::CONNECT;
			}
			else {
				blocks[x][y].setBoundaryType(BND_TOP, scenario.getBoundaryType(BND_TOP));
			}
			if (y != nBlocksY - 1) {
				blocks[x][y].neighbours[BND_BOTTOM] = &blocks[x][y + 1];
				blocks[x][y].boundary[BND_BOTTOM] = BoundaryType::CONNECT;
			}
			else {
				blocks[x][y].setBoundaryType(BND_BOTTOM, scenario.getBoundaryType(BND_BOTTOM));
			}
		}
	}
}

SWE_StarPU_Sim::~SWE_StarPU_Sim()
{
	starpu_data_unregister(spu_timestamp);
	starpu_data_unregister(spu_nextTimestampToWrite);
	starpu_data_unregister(spu_maxTimestep);
	for (size_t x = 0; x < scratchData.size(); ++x) {
		for (size_t y = 0; y < scratchData[x].size(); ++y) {
			starpu_data_unregister(scratchData[x][y]);
			blocks[x][y].starpu_unregister();
		}
	}
	scratchData.clear();
}

void SWE_StarPU_Sim::launchTaskGraph()
{
	writeTimeStep();
	runTimestep();
}

void SWE_StarPU_Sim::writeTimeStep()
{
	if (getenv("SWE_DISABLE_WRITE")) {
		return;
	}
	for (size_t x = 0; x < writers.size(); ++x) {
		for (size_t y = 0; y < writers[x].size(); ++y) {
			const auto pWriters = &writers[x][y];
			starpu_task_insert(&SWECodelets::resultWriter,
				(STARPU_PRIORITY), 1,
				STARPU_VALUE, &pWriters, sizeof(pWriters),
				STARPU_R, blocks[x][y].huvData().starpuHandle(),
				STARPU_R, blocks[x][y].bStarpuHandle(),
				STARPU_R, spu_timestamp,
				0);
		}
	}
}

void SWE_StarPU_Sim::updateGhostLayers()
{
	for (size_t x = 0; x < blocks.size(); ++x) {
		for (size_t y = 0; y < blocks[x].size(); ++y) {
			auto side = BND_LEFT;
			auto blockptr = &blocks[x][y];

			starpu_task_insert(&SWECodelets::updateGhostLayers,
				STARPU_PRIORITY, 0,
				STARPU_VALUE, &side, sizeof(side),
				STARPU_VALUE, &blockptr, sizeof(blockptr),
				STARPU_W, blocks[x][y].boundaryData[side].starpuHandle(),
				STARPU_R, blocks[x][y].huvData().starpuHandle(),
				STARPU_R, blocks[x][y].neighbours[side] ? blocks[x][y].neighbours[side]->huvData().starpuHandle() : blocks[x][y].huvData().starpuHandle(),
				0);
			side = BND_RIGHT;
			starpu_task_insert(&SWECodelets::updateGhostLayers,
				STARPU_PRIORITY, 0,
				STARPU_VALUE, &side, sizeof(side),
				STARPU_VALUE, &blockptr, sizeof(blockptr),
				STARPU_W, blocks[x][y].boundaryData[side].starpuHandle(),
				STARPU_R, blocks[x][y].huvData().starpuHandle(),
				STARPU_R, blocks[x][y].neighbours[side] ? blocks[x][y].neighbours[side]->huvData().starpuHandle() : blocks[x][y].huvData().starpuHandle(),
				0);
			side = BND_BOTTOM;
			starpu_task_insert(&SWECodelets::updateGhostLayers,
				STARPU_PRIORITY, 0,
				STARPU_VALUE, &side, sizeof(side),
				STARPU_VALUE, &blockptr, sizeof(blockptr),
				STARPU_W, blocks[x][y].boundaryData[side].starpuHandle(),
				STARPU_R, blocks[x][y].huvData().starpuHandle(),
				STARPU_R, blocks[x][y].neighbours[side] ? blocks[x][y].neighbours[side]->huvData().starpuHandle() : blocks[x][y].huvData().starpuHandle(),
				0);
			side = BND_TOP;
			starpu_task_insert(&SWECodelets::updateGhostLayers,
				STARPU_PRIORITY, 0,
				STARPU_VALUE, &side, sizeof(side),
				STARPU_VALUE, &blockptr, sizeof(blockptr),
				STARPU_W, blocks[x][y].boundaryData[side].starpuHandle(),
				STARPU_R, blocks[x][y].huvData().starpuHandle(),
				STARPU_R, blocks[x][y].neighbours[side] ? blocks[x][y].neighbours[side]->huvData().starpuHandle() : blocks[x][y].huvData().starpuHandle(),
				0);
		}
	}
}

void SWE_StarPU_Sim::computeNumericalFluxes()
{
	for (size_t x = 0; x < blocks.size(); ++x) {
		for (size_t y = 0; y < blocks[x].size(); ++y) {
			const auto blockptr = &blocks[x][y];
			starpu_task_insert(&SWECodelets::computeNumericalFluxes,
				STARPU_PRIORITY, 0,
				STARPU_VALUE, &blockptr, sizeof(blockptr),
				STARPU_R, blocks[x][y].huvData().starpuHandle(),
				STARPU_R, blocks[x][y].boundaryData[BND_LEFT].starpuHandle(),
				STARPU_R, blocks[x][y].boundaryData[BND_RIGHT].starpuHandle(),
				STARPU_R, blocks[x][y].boundaryData[BND_BOTTOM].starpuHandle(),
				STARPU_R, blocks[x][y].boundaryData[BND_TOP].starpuHandle(),
				STARPU_R, blocks[x][y].bStarpuHandle(),
				STARPU_W, scratchData[x][y],
				STARPU_REDUX, spu_maxTimestep,
				0);
		}
	}
}

void SWE_StarPU_Sim::updateUnkowns()
{
	for (size_t x = 0; x < blocks.size(); ++x) {
		for (size_t y = 0; y < blocks[x].size(); ++y) {
			const auto blockptr = &blocks[x][y];
			starpu_task_insert(&SWECodelets::updateUnknowns,
				STARPU_PRIORITY, 0,
				STARPU_VALUE, &blockptr, sizeof(blockptr),
				STARPU_RW, blocks[x][y].huvData().starpuHandle(),
				STARPU_R, scratchData[x][y],
				STARPU_R, spu_maxTimestep,
				0);
		}
	}
}

void SWE_StarPU_Sim::runTimestep()
{
	starpu_iteration_push(iterationNumber);
	updateGhostLayers();
	computeNumericalFluxes();
	updateUnkowns();

	const auto pSim = this;
	const auto pCheckpoints = &l_checkPoints;

	starpu_task_insert(&SWECodelets::incrementTime,
		STARPU_PRIORITY, 3,
		STARPU_VALUE, &pSim, sizeof(pSim),
		STARPU_VALUE, &pCheckpoints, sizeof(pCheckpoints),
		STARPU_RW, spu_timestamp,
		STARPU_R, spu_maxTimestep,
		STARPU_RW, spu_nextTimestampToWrite,
		0);
	starpu_iteration_pop();
	iterationNumber++;
}
