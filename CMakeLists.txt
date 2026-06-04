cmake_minimum_required(VERSION 3.18)
project(HydraEngine)

find_package(Torch REQUIRED)

# Aggressive C++ optimization flags for speed
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2 -ffast-math")

add_executable(hydra_native cpp/engine.cpp)
target_link_libraries(hydra_native "${TORCH_LIBRARIES}")
set_property(TARGET hydra_native PROPERTY CXX_STANDARD 20)
