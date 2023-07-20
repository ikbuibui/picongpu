
#include <pmacc/lockstep.hpp>
#include <pmacc/mappings/threads/ThreadCollective.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/memory/boxes/CachedBox.hpp>
// #include <pmacc/memory/dataTypes/Mask.hpp>
// #include <pmacc/random/Random.hpp>
// #include <pmacc/random/distributions/distributions.hpp>
// #include <pmacc/random/methods/methods.hpp>
// #include "pmacc/dimensions/DataSpaceOperations.hpp"
// #include <pmacc/mappings/kernel/AreaMapping.hpp>

struct HelloKernel
{
    /** run stencil for a supercell
     *
     * @tparam T_Box PMacc::DataBox, box type 
     * @tparam T_Mapping mapping functor type
     *
     * @param buff databox of the buffer
     * @param mapper functor to map a block to a supercell
     */
    template <typename T_Box, typename T_Mapping, typename T_Worker>
    DINLINE void operator()(
        T_Worker const &worker,
        T_Box const &buff,
        uint32_t const &globalRank,
        pmacc::DataSpace<DIM2> const &localOffset,
        T_Mapping const &mapper) const
    {
        using Type = typename T_Box::ValueType;
        using SuperCellSize = typename T_Mapping::SuperCellSize;
        using BlockArea = pmacc::SuperCellDescription<SuperCellSize, pmacc::math::CT::Int<1, 1>, pmacc::math::CT::Int<1, 1>>;

        pmacc::DataSpace<DIM2> const block(mapper.getSuperCellIndex(pmacc::DataSpace<DIM2>(cupla::blockIdx(worker.getAcc()))));
        pmacc::DataSpace<DIM2> const blockCell = block * T_Mapping::SuperCellSize::toRT();

        // constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

        // // what is this?
        // auto cache = pmacc::CachedBox::create<0, Type>(worker, BlockArea());
        // auto buff_shifted = buff.shift(blockCell);

        // auto collective = pmacc::makeThreadCollective<BlockArea>();

        // pmacc::math::operation::Assign assign;
        // collective(worker, assign, cache, buff_shifted);

        // worker.sync();

        pmacc::lockstep::makeForEach<pmacc::math::CT::volume<SuperCellSize>::type::value>(worker)(
            [&](uint32_t const linearIdx) {
                // cell index within the superCell
                pmacc::DataSpace<DIM2> const cellIdx =
                    pmacc::DataSpaceOperations<DIM2>::template map<SuperCellSize>(linearIdx);
                auto globalPos = cellIdx + blockCell + localOffset;
                printf("Hello World from rank %u, global position %d %d,  cellIdx %d %d, and block index %d %d \n", globalRank, globalPos[0], globalPos[1], cellIdx[0], cellIdx[1], blockCell[0], blockCell[1]);
            });
    };
};