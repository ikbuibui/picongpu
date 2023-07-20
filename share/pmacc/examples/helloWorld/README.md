## Installing
- Make a build directory
- cmake ..
- ccmake .
    - choose accelerator backend and configure
- cmake --build .

## Running
- mpirun -npernode 4 -n 4 ./helloWorld
    - make sure that -npernode and -n are set according to the number of devices in the code, when you initialize the pmacc::Environment
