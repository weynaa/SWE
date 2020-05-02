SWE-StarPU
===

The Shallow Water Equations teaching code. This fork implements prototypes for targeting the [StarPU](https://starpu.gforge.inria.fr/) framework as part of the Future trends in High Performance Computing Seminar 2020.

[StarPU-Documentation](http://starpu.gforge.inria.fr/testing/master/doc/html/)

Documentation
-------------

The documentation is available in the [Wiki](https://github.com/TUM-I5/SWE/wiki)

Building 
------------
For debian based distributions, install the StarPU-packages
```shell script
$ sudo apt-get install libstarpu-1.3 libstarpu-dev
```
If StarPU is successfully installed, check your available resources:
 ```shell script
$ starpu_machine_display
 ```

With StarPU installed, run the CMake-build:
 ```shell script
SWE$ mkdir build
SWE$ cd build
SWE/build$ cmake ..
SWE/build$ make -j 8
SWE/build$ ./swe-starpu -x 1000 -y 1000 -o testoutput
 ```


License
-------

SWE is release unter GPLv3 (see [gpl.txt](gpl.txt))
