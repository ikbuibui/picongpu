/* Standalone helper test for the field-absorber selection policy.
 *
 * Compiled twice by CMake: once with PICONGPU_MINIMAL_CARAVAN_THERMAL defined
 * and once without. This exercises the pure policy helpers; the production
 * AbsorberFactory integration is covered by absorberFactoryTest.cpp.
 */

#include "picongpu/fields/absorber/AbsorberPolicy.hpp"

#include <array>
#include <cstdlib>
#include <initializer_list>
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

    bool effectiveThrows(
        AbsorberKind const requested,
        std::array<bool, 3> const& periodic,
        uint32_t const activeDimensions,
        std::string* const message = nullptr)
    {
        try
        {
            static_cast<void>(effectiveAbsorberKind(requested, periodic, activeDimensions));
            return false;
        }
        catch(std::runtime_error const& error)
        {
            if(message != nullptr)
                *message = error.what();
            return true;
        }
    }

    std::array<bool, 3> withNonPeriodic(uint32_t const axis)
    {
        std::array<bool, 3> periodic{true, true, true};
        periodic[axis] = false;
        return periodic;
    }
} // namespace

int main()
{
    std::array<bool, 3> const allPeriodic3D{true, true, true};
    // In 2D the z flag is false but inactive; it must not influence the policy.
    std::array<bool, 3> const periodic2DInactiveZ{true, true, false};

    std::string message;

#if defined(PICONGPU_MINIMAL_CARAVAN_THERMAL)
    // All active (3D or 2D) dimensions periodic resolve to None for every requested kind.
    for(auto const requested : {AbsorberKind::None, AbsorberKind::Pml, AbsorberKind::Exponential})
    {
        check(
            effectiveAbsorberKind(requested, allPeriodic3D, 3) == AbsorberKind::None,
            "minimal 3D all-periodic must resolve to None");
        check(
            effectiveAbsorberKind(requested, periodic2DInactiveZ, 2) == AbsorberKind::None,
            "minimal 2D all-active-periodic must resolve to None and ignore inactive z");
    }

    // Exercise each active axis independently, with requested None, and verify the reason.
    for(uint32_t axis = 0u; axis < 3u; ++axis)
    {
        message.clear();
        check(
            effectiveThrows(AbsorberKind::None, withNonPeriodic(axis), 3, &message),
            "minimal non-periodic active axis must be rejected (requested None)");
        check(
            message.find("requires periodic boundaries") != std::string::npos,
            "minimal rejection must report the periodicity policy");
    }
    check(
        effectiveThrows(AbsorberKind::None, std::array<bool, 3>{false, true, false}, 2),
        "minimal 2D non-periodic x must be rejected");
    check(
        effectiveThrows(AbsorberKind::None, std::array<bool, 3>{true, false, false}, 2),
        "minimal 2D non-periodic y must be rejected");

    // Factory-kind helper policy.
    check(!effectiveThrows(AbsorberKind::None, allPeriodic3D, 3), "minimal None must be accepted");
    check(effectiveThrows(static_cast<AbsorberKind>(999), allPeriodic3D, 3), "invalid enum must be rejected");
#else
    for(auto const requested : {AbsorberKind::None, AbsorberKind::Pml, AbsorberKind::Exponential})
        check(
            effectiveAbsorberKind(requested, allPeriodic3D, 3) == AbsorberKind::None,
            "ordinary 3D all-periodic must resolve to None");
    check(
        effectiveAbsorberKind(AbsorberKind::Pml, periodic2DInactiveZ, 2) == AbsorberKind::None,
        "ordinary 2D all-active-periodic must resolve to None and ignore inactive z");

    check(
        effectiveAbsorberKind(AbsorberKind::Pml, withNonPeriodic(0), 3) == AbsorberKind::Pml,
        "ordinary non-periodic x must keep requested Pml");
    check(
        effectiveAbsorberKind(AbsorberKind::Exponential, withNonPeriodic(1), 3) == AbsorberKind::Exponential,
        "ordinary non-periodic y must keep requested Exponential");
    check(
        effectiveAbsorberKind(AbsorberKind::None, withNonPeriodic(2), 3) == AbsorberKind::None,
        "ordinary non-periodic z with requested None stays None");
    check(effectiveThrows(static_cast<AbsorberKind>(999), allPeriodic3D, 3), "invalid enum must be rejected");
#endif

    if(failures != 0)
    {
        std::cerr << failures << " absorber-policy helper check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "absorber-policy helper checks passed\n";
    return EXIT_SUCCESS;
}
