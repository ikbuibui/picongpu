/* Standalone factory integration test for the field-absorber policy.
 *
 * Invokes the production AbsorberFactory::setKind()/getKind() to verify that
 * rejected kinds cannot be installed. Built in minimal and ordinary variants.
 */

#include "picongpu/fields/absorber/AbsorberImpl.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace picongpu::fields::absorber;

namespace
{
    int failures = 0;

    void check(bool const condition, std::string const& message)
    {
        if(!condition)
        {
            std::cerr << "FAIL: " << message << "\n";
            ++failures;
        }
    }

    bool setKindThrows(AbsorberFactory& factory, AbsorberKind const kind)
    {
        try
        {
            factory.setKind(kind);
            return false;
        }
        catch(std::runtime_error const&)
        {
            return true;
        }
    }
} // namespace

int main()
{
    auto& factory = AbsorberFactory::get();

#if defined(PICONGPU_MINIMAL_CARAVAN_THERMAL)
    factory.setKind(AbsorberKind::None);
    check(factory.getKind() == AbsorberKind::None, "minimal factory must install None");

    check(setKindThrows(factory, AbsorberKind::Pml), "minimal factory must reject Pml");
    check(factory.getKind() == AbsorberKind::None, "rejected Pml must not change installed kind");

    check(setKindThrows(factory, AbsorberKind::Exponential), "minimal factory must reject Exponential");
    check(factory.getKind() == AbsorberKind::None, "rejected Exponential must not change installed kind");

    check(
        setKindThrows(factory, static_cast<AbsorberKind>(999)),
        "minimal factory must reject an invalid enum value");
    check(factory.getKind() == AbsorberKind::None, "rejected invalid value must not change installed kind");
#else
    factory.setKind(AbsorberKind::Exponential);
    check(factory.getKind() == AbsorberKind::Exponential, "ordinary factory must install Exponential");

    factory.setKind(AbsorberKind::Pml);
    check(factory.getKind() == AbsorberKind::Pml, "ordinary factory must install Pml");

    factory.setKind(AbsorberKind::None);
    check(factory.getKind() == AbsorberKind::None, "ordinary factory must install None");

    check(
        setKindThrows(factory, static_cast<AbsorberKind>(999)),
        "ordinary factory must reject an invalid enum value");
    check(factory.getKind() == AbsorberKind::None, "rejected invalid value must not change installed kind");
#endif

    if(failures != 0)
    {
        std::cerr << failures << " absorber-factory check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "absorber-factory checks passed\n";
    return EXIT_SUCCESS;
}
