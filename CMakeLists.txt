
cmake_minimum_required(VERSION 3.10.0)

# specify the project name and language
project(cosyr LANGUAGES CXX)

# policy for <PackageName>_ROOT variables
cmake_policy(SET CMP0074 NEW)

# create the target
add_executable(cosyr)

# set compiler standard and extensions
target_compile_features(cosyr PRIVATE cxx_std_14)
set(CMAKE_CXX_EXTENSIONS OFF)

#------------------------------------------------------------------------------#
# Enable Pybind11
#------------------------------------------------------------------------------#
find_package(pybind11 REQUIRED HINTS ${pybind11_DIR}/share/cmake)  # or `add_subdirectory(pybind11)
target_link_libraries(cosyr PRIVATE pybind11::embed)

#------------------------------------------------------------------------------#
# Enable OpenMP and MPI
#------------------------------------------------------------------------------#
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
target_link_libraries(cosyr PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(cosyr PRIVATE MPI::MPI_CXX)

#------------------------------------------------------------------------------#
# Enable Portage
#------------------------------------------------------------------------------#
find_package(portage REQUIRED HINTS ${PORTAGE_DIR}/lib/cmake)
target_link_libraries(cosyr PRIVATE portage::portage)

#-----------------------------------------------------------------------------
# Enable Kokkos
#-----------------------------------------------------------------------------
find_package(Kokkos REQUIRED HINTS ${Kokkos_DIR})
target_link_libraries(cosyr PRIVATE Kokkos::kokkos)
target_compile_definitions(cosyr PRIVATE ENABLE_KOKKOS=1)

#-----------------------------------------------------------------------------
# Enable Cabana 
#-----------------------------------------------------------------------------
find_package(Cabana REQUIRED HINTS ${Cabana_DIR})
target_link_libraries(cosyr PRIVATE Cabana::cabanacore)
target_compile_definitions(cosyr PRIVATE ENABLE_CABANA=1)

#-----------------------------------------------------------------------------
# Enable TCMalloc
#-----------------------------------------------------------------------------
if (ENABLE_TCMALLOC)
  string(APPEND CMAKE_EXE_LINKER_FLAGS " -ltcmalloc ${TCMALLOC_LIB}")
endif()

# disable GCC visibility warnings
target_compile_options(cosyr PRIVATE "-Wno-attributes")

# set headers source and libraries
target_include_directories(cosyr PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_sources(cosyr PRIVATE
    src/main.cpp
    src/beam.cpp
    src/mesh.cpp
    src/remap.cpp
    src/pusher.cpp
    src/wavelet.cpp
    src/input.cpp
    src/io.cpp
    src/formula.cpp
    src/analysis.cpp)
