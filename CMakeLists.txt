cmake_minimum_required(VERSION 3.18)
project(HydraEngine LANGUAGES CXX)

# Find required packages
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)

# C++17 required for structured bindings and constexpr if
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# AGGRESSIVE OPTIMIZATION FLAGS
# ============================================================

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Base flags
add_compile_definitions(NDEBUG)

if(MSVC)
    # Windows MSVC: maximum optimization
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Oi /Ot /GL /arch:AVX2 /fp:fast /GS-")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG /OPT:REF /OPT:ICF")
    add_compile_definitions(_USE_MATH_DEFINES NOMINMAX)
else()
    # Linux GCC/Clang: maximum optimization
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -ffast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -funroll-loops -fprefetch-loop-arrays")
    
    # Check for AVX2
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mbmi -mbmi2")
        message(STATUS "[HYDRA] AVX2 + FMA + BMI enabled")
    endif()
    
    # LTO: cross-file inlining and dead code elimination
    check_cxx_compiler_flag("-flto" COMPILER_SUPPORTS_LTO)
    if(COMPILER_SUPPORTS_LTO)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=auto")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=auto")
        message(STATUS "[HYDRA] Link-time optimization enabled")
    endif()
    
    # Profile-guided optimization hint
    check_cxx_compiler_flag("-fprofile-use" COMPILER_SUPPORTS_PGO)
    if(COMPILER_SUPPORTS_PGO)
        message(STATUS "[HYDRA] PGO available: build with -DCMAKE_CXX_FLAGS=-fprofile-generate, run, then rebuild with -fprofile-use")
    endif()
endif()

# Re-enable exceptions for engine.cpp (it uses try/catch for model loading)
# But sampling.cpp and extension.cpp can be exception-free
if(NOT MSVC)
    set_source_files_properties(cpp/engine.cpp PROPERTIES COMPILE_FLAGS "-fexceptions")
endif()

# ============================================================
# BUILD TARGET
# ============================================================
add_executable(hydra_native 
    cpp/engine.cpp 
    cpp/sampling.cpp 
    cpp/extension.cpp
)

target_link_libraries(hydra_native "${TORCH_LIBRARIES}")
target_include_directories(hydra_native PRIVATE ${TORCH_INCLUDE_DIRS} cpp)

# Link CUDA runtime for CUDA Graph API and events
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    target_link_libraries(hydra_native CUDA::cudart)
    message(STATUS "[HYDRA] CUDA Runtime linked for CUDA Graph support")
endif()

# ============================================================
# BUILD SUMMARY
# ============================================================
message(STATUS "==============================================")
message(STATUS "  HYDRA ENGINE - BUILD CONFIGURATION")
message(STATUS "==============================================")
message(STATUS "  Compiler:    ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build Type:  ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX Flags:   ${CMAKE_CXX_FLAGS}")
message(STATUS "  CUDA Toolkit:${CUDAToolkit_FOUND}")
message(STATUS "==============================================")
