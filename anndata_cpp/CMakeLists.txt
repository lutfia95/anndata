cmake_minimum_required(VERSION 3.20)

project(anndata_cpp LANGUAGES CXX)

set(HDF5_SERIAL_LIBRARY "/usr/lib/x86_64-linux-gnu/libhdf5_serial.so.103")
if(NOT EXISTS "${HDF5_SERIAL_LIBRARY}")
    message(FATAL_ERROR "Expected HDF5 runtime at ${HDF5_SERIAL_LIBRARY}")
endif()

add_library(
    anndata_cpp
    src/anndata.cpp
)

target_include_directories(
    anndata_cpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(anndata_cpp PUBLIC cxx_std_17)
target_compile_options(anndata_cpp PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(anndata_cpp PUBLIC ${HDF5_SERIAL_LIBRARY})

add_executable(
    anndata_cpp_main
    src/main.cpp
)

target_link_libraries(anndata_cpp_main PRIVATE anndata_cpp ${HDF5_SERIAL_LIBRARY})
target_compile_features(anndata_cpp_main PRIVATE cxx_std_17)
target_compile_options(anndata_cpp_main PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()

add_executable(
    anndata_cpp_tests
    tests/test_main.cpp
)

target_link_libraries(anndata_cpp_tests PRIVATE anndata_cpp ${HDF5_SERIAL_LIBRARY})
target_include_directories(anndata_cpp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(anndata_cpp_tests PRIVATE cxx_std_17)
target_compile_options(anndata_cpp_tests PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(
    anndata_cpp_tests
    PRIVATE ANNDATA_CPP_REPO_ROOT="${CMAKE_SOURCE_DIR}/.."
)

add_test(NAME anndata_cpp_tests COMMAND anndata_cpp_tests)
