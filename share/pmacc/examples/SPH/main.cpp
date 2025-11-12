#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/Definition.hpp>

#include <iostream>

int main()
{
    auto grid = pmacc::MemSpace<DIM2>::create(10);

    std::cout << "hello SPH! grid size is " << grid.x() << "\t" << grid.y() << std::endl;
    return 0;
}
