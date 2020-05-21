#ifndef SWE_STARPUBLOCKWRITER_H
#define SWE_STARPUBLOCKWRITER_H
#if defined(ENABLE_STARPU)

#include <string>
#include <netcdf.h>
#include <starpu/SWE_HUV_Matrix.h>

namespace io {
    class StarPUBlockWriter {
        const std::string fileName;
        size_t timeStep = 0;
        const float originX, originY;
        const float dX, dY;

        int timeVar, hVar, huVar, hvVar, bVar;
        int dataFile;

        size_t nX, nY;

        /** Flush after every x write operation? */
        unsigned int flush;

    public:
        StarPUBlockWriter(const std::string &_fileName,
                          size_t _nX, size_t _nY,
                          float _dX, float _dY,
                          float _originX, float _originY) :
                fileName(_fileName + ".nc"),
                originX(_originX), originY(_originY),
                dX(_dX), dY(_dY), nX(_nX), nY(_nY) {
            int status;

            //create a netCDF-file, an existing file will be replaced
            status = nc_create(fileName.c_str(), NC_NETCDF4, &dataFile);

            //check if the netCDF-file creation constructor succeeded.
            if (status != NC_NOERR) {
                assert(false);
                return;
            }
            int l_timeDim, l_xDim, l_yDim;
            nc_def_dim(dataFile, "time", NC_UNLIMITED, &l_timeDim);
            nc_def_dim(dataFile, "x", nX, &l_xDim);
            nc_def_dim(dataFile, "y", nY, &l_yDim);

            //variables (TODO: add rest of CF-1.5)
            int l_xVar, l_yVar;

            nc_def_var(dataFile, "time", NC_FLOAT, 1, &l_timeDim, &timeVar);
            ncPutAttText(timeVar, "long_name", "Time");
            ncPutAttText(timeVar, "units",
                         "seconds since simulation start"); // the word "since" is important for the paraview reader

            nc_def_var(dataFile, "x", NC_FLOAT, 1, &l_xDim, &l_xVar);
            nc_def_var(dataFile, "y", NC_FLOAT, 1, &l_yDim, &l_yVar);

            //variables, fastest changing index is on the right (C syntax), will be mirrored by the library
            int dims[] = {l_timeDim, l_yDim, l_xDim};
            nc_def_var(dataFile, "h", NC_FLOAT, 3, dims, &hVar);
            nc_def_var(dataFile, "hu", NC_FLOAT, 3, dims, &huVar);
            nc_def_var(dataFile, "hv", NC_FLOAT, 3, dims, &hvVar);
            nc_def_var(dataFile, "b", NC_FLOAT, 2, &dims[1], &bVar);
            ncPutAttText(NC_GLOBAL, "Conventions", "CF-1.5");
            ncPutAttText(NC_GLOBAL, "title", "Computed tsunami solution");
            ncPutAttText(NC_GLOBAL, "history", "SWE");
            ncPutAttText(NC_GLOBAL, "institution",
                         "Technische Universitaet Muenchen, Department of Informatics, Chair of Scientific Computing");
            ncPutAttText(NC_GLOBAL, "source", "Bathymetry and displacement data.");
            ncPutAttText(NC_GLOBAL, "references", "http://www5.in.tum.de/SWE");
            ncPutAttText(NC_GLOBAL, "comment",
                         "SWE is free software and licensed under the GNU General Public License. Remark: In general this does not hold for the used input data.");
            //setup grid size
            float gridPosition = originX + (float) .5 * dX;
            for (size_t i = 0; i < nX; i++) {
                nc_put_var1_float(dataFile, l_xVar, &i, &gridPosition);

                gridPosition += dX;
            }

            gridPosition = originY + (float) .5 * dY;
            for (size_t j = 0; j < nY; j++) {
                nc_put_var1_float(dataFile, l_yVar, &j, &gridPosition);
                gridPosition += dY;
            }
        }

        ~StarPUBlockWriter() {
            nc_close(dataFile);
        }

        void writeTimeStep(SWE_HUV_Matrix_interface &huvData, starpu_matrix_interface &b, float time) {
            if (timeStep == 0)
                // Write bathymetry
                writeVarTimeIndependent(b, bVar,1,1);

            //write i_time
            nc_put_var1_float(dataFile, timeVar, &timeStep, &time);

            //write water height
            writeVarTimeDependent(huvData.h, huvData.nX, huvData.ld, huvData.nY, hVar);

            //write momentum in x-direction
            writeVarTimeDependent(huvData.hu, huvData.nX, huvData.ld, huvData.nY, huVar);

            //write momentum in y-direction
            writeVarTimeDependent(huvData.hv, huvData.nX, huvData.ld, huvData.nY, hvVar);

            // Increment timeStep for next call
            timeStep++;

            if (flush > 0 && timeStep % flush == 0)
                nc_sync(dataFile);
        }

    private:
        void writeVarTimeIndependent(starpu_matrix_interface &i_matrix,
                                     int i_ncVariable,
                                     const size_t xOffsset, const size_t yOffset) const {
            //write col wise, necessary to get rid of the boundary
            //storage in Float2D is col wise
            //read carefully, the dimensions are confusing
            size_t start[] = {0, 0};
            size_t count[] = {1, nX};
            for (unsigned int row = 0; row < nY; row++) {
                start[0] = row; //select row (dim "y")
                nc_put_vara_float(dataFile, i_ncVariable, start, count,
                                  &(((float_type *) i_matrix.ptr)[(row+yOffset) * i_matrix.ld+xOffsset])); //write row
            }
        }

        template<typename T>
        void writeVarTimeDependent(const T *data, const size_t nX, const size_t ld,
                                   const size_t nY,
                                   int i_ncVariable) const {
            //write col wise, necessary to get rid of the boundary
            //storage in Float2D is col wise
            //read carefully, the dimensions are confusing
            size_t start[] = {timeStep, 0, 0};
            size_t count[] = {1, 1, nX};
            for (unsigned int row = 0; row < nY; row++) {
                start[1] = row; //select row (dim "x")
                nc_put_vara_float(dataFile, i_ncVariable, start, count,
                                  &data[row * ld]); //write row
            }
        }

        void ncPutAttText(int varid, const char *name, const char *value) const {
            nc_put_att_text(dataFile, varid, name, strlen(value), value);
        }
    };
}
#endif
#endif //SWE_STARPUBLOCKWRITER_H
