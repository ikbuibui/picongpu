
#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/resource/resource_user.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Resource User")
{
    auto rg = redGrapes::init(1);

    redGrapes::IOResource<int> a, b;

    redGrapes::ResourceUser f1({a.read()}, 0, 0);
    redGrapes::ResourceUser f2({a.read(), a.write()}, 0, 0);
    redGrapes::ResourceUser f3({b.read()}, 0, 0);
    redGrapes::ResourceUser f4({b.read(), b.write()}, 0, 0);
    redGrapes::ResourceUser f5({a.read(), a.write(), b.read(), b.write()}, 0, 0);

    REQUIRE(is_serial(f1, f1) == false);
    REQUIRE(is_serial(f1, f2) == true);
    REQUIRE(is_serial(f1, f3) == false);
    REQUIRE(is_serial(f1, f4) == false);
    REQUIRE(is_serial(f1, f5) == true);

    REQUIRE(is_serial(f2, f1) == true);
    REQUIRE(is_serial(f2, f2) == true);
    REQUIRE(is_serial(f2, f3) == false);
    REQUIRE(is_serial(f2, f4) == false);
    REQUIRE(is_serial(f2, f5) == true);

    REQUIRE(is_serial(f3, f1) == false);
    REQUIRE(is_serial(f3, f2) == false);
    REQUIRE(is_serial(f3, f3) == false);
    REQUIRE(is_serial(f3, f4) == true);
    REQUIRE(is_serial(f3, f5) == true);

    REQUIRE(is_serial(f4, f1) == false);
    REQUIRE(is_serial(f4, f2) == false);
    REQUIRE(is_serial(f4, f3) == true);
    REQUIRE(is_serial(f4, f4) == true);
    REQUIRE(is_serial(f4, f5) == true);

    REQUIRE(is_serial(f5, f1) == true);
    REQUIRE(is_serial(f5, f2) == true);
    REQUIRE(is_serial(f5, f3) == true);
    REQUIRE(is_serial(f5, f4) == true);
    REQUIRE(is_serial(f5, f5) == true);


    REQUIRE(f1.is_superset_of(f1) == true);
    REQUIRE(f1.is_superset_of(f2) == false);
    REQUIRE(f1.is_superset_of(f3) == false);
    REQUIRE(f1.is_superset_of(f4) == false);
    REQUIRE(f1.is_superset_of(f5) == false);

    REQUIRE(f2.is_superset_of(f1) == true);
    REQUIRE(f2.is_superset_of(f2) == true);
    REQUIRE(f2.is_superset_of(f3) == false);
    REQUIRE(f2.is_superset_of(f4) == false);
    REQUIRE(f2.is_superset_of(f5) == false);
}
