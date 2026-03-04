cmake_minimum_required(VERSION 3.16)
project(OptionsEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
endif()

if(MSVC)
    add_compile_options(/W4 /permissive-)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
endif()

find_package(Threads REQUIRED)

add_library(options_engine
    src/math_utils.cpp
    src/black_scholes.cpp
    src/monte_carlo.cpp
    src/implied_vol.cpp
    src/binomial.cpp
    src/heston.cpp
)

target_include_directories(options_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(options_engine PUBLIC Threads::Threads)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_all tests/test_all.cpp)
target_link_libraries(test_all PRIVATE options_engine GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_all)
