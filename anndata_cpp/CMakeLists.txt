cmake_minimum_required(VERSION 3.20)

project(anndata_cpp LANGUAGES CXX)

find_library(
    ANNDATA_CPP_HDF5_LIBRARY
    NAMES hdf5_serial hdf5 libhdf5_serial.so.103 libhdf5.so
    PATHS /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/local/lib
)
if(NOT ANNDATA_CPP_HDF5_LIBRARY)
    message(FATAL_ERROR "Could not find an HDF5 runtime library (looked for hdf5_serial and hdf5)")
endif()

option(ANNDATA_CPP_ENABLE_DOXYGEN "Enable the Doxygen documentation target when Doxygen is available" ON)

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
target_link_libraries(anndata_cpp PUBLIC ${ANNDATA_CPP_HDF5_LIBRARY})

add_executable(
    anndata_cpp_main
    src/main.cpp
)

target_link_libraries(anndata_cpp_main PRIVATE anndata_cpp ${ANNDATA_CPP_HDF5_LIBRARY})
target_compile_features(anndata_cpp_main PRIVATE cxx_std_17)
target_compile_options(anndata_cpp_main PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()

add_executable(
    anndata_cpp_tests
    tests/test_main.cpp
)

target_link_libraries(anndata_cpp_tests PRIVATE anndata_cpp ${ANNDATA_CPP_HDF5_LIBRARY})
target_include_directories(anndata_cpp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(anndata_cpp_tests PRIVATE cxx_std_17)
target_compile_options(anndata_cpp_tests PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(
    anndata_cpp_tests
    PRIVATE ANNDATA_CPP_REPO_ROOT="${CMAKE_SOURCE_DIR}/.."
)

add_test(NAME anndata_cpp_tests COMMAND anndata_cpp_tests)

if(ANNDATA_CPP_ENABLE_DOXYGEN)
    find_package(Doxygen QUIET)
    if(DOXYGEN_FOUND)
        set(ANNDATA_CPP_DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/doxygen")
        set(ANNDATA_CPP_DOXYFILE "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")

        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
            ${ANNDATA_CPP_DOXYFILE}
            @ONLY
        )

        add_custom_target(
            anndata_cpp_docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${ANNDATA_CPP_DOXYFILE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating Doxygen documentation for anndata_cpp"
            VERBATIM
        )
    else()
        message(STATUS "Doxygen not found; skipping anndata_cpp_docs target")
    endif()
endif()
