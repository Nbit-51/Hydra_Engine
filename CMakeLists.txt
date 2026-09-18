cmake_minimum_required(VERSION 3.18)
project(HydraEngine LANGUAGES CXX)

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_definitions(NDEBUG)

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Oi /Ot /GL /fp:fast /GS-")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG /OPT:REF /OPT:ICF")
    add_compile_definitions(_USE_MATH_DEFINES NOMINMAX)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -ffast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -funroll-loops -fprefetch-loop-arrays")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=auto")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=auto")
endif()

set_source_files_properties(cpp/engine.cpp PROPERTIES COMPILE_FLAGS "-fexceptions")

add_executable(hydra_native cpp/engine.cpp)
target_link_libraries(hydra_native "${TORCH_LIBRARIES}")
target_include_directories(hydra_native PRIVATE ${TORCH_INCLUDE_DIRS} cpp)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    target_link_libraries(hydra_native CUDA::cudart)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(
        hydra_simd_test
        tests/cpp/test_simd_verify.cpp
        cpp/simd_verify.cpp
    )
    target_include_directories(hydra_simd_test PRIVATE cpp)
    if(MSVC)
        target_compile_options(hydra_simd_test PRIVATE /arch:AVX2)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        target_compile_options(hydra_simd_test PRIVATE -mavx2)
    endif()
    add_test(NAME hydra_simd_verify COMMAND hydra_simd_test)
endif()
