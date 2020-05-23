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
                          float _originX, float _originY, unsigned int _flush);

        ~StarPUBlockWriter() {
            nc_close(dataFile);
        }

        void writeTimeStep(SWE_HUV_Matrix_interface &huvData, starpu_matrix_interface &b, float time);

    private:
        void writeVarTimeIndependent(starpu_matrix_interface &i_matrix,
                                     int i_ncVariable,
                                     const size_t xOffsset, const size_t yOffset) const;

        template<typename T>
        void writeVarTimeDependent(const T *data, const size_t _nX, const size_t ld,
                                   const size_t _nY,
                                   int i_ncVariable) const {
            //write col wise, necessary to get rid of the boundary
            //storage in Float2D is col wise
            //read carefully, the dimensions are confusing
            size_t start[] = {timeStep, 0, 0};
            size_t count[] = {1, 1, _nX};
            for (unsigned int row = 0; row < _nY; row++) {
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
