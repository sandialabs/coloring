cmake_minimum_required(VERSION 3.22.0)

project(coloring VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(coloring
    lodepng/lodepng.cpp
    coloring.cpp
    )
target_compile_features(coloring PUBLIC cxx_std_17)
target_include_directories(coloring PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/lodepng")
set_target_properties(coloring PROPERTIES
  PUBLIC_HEADER coloring.hpp)

# workaround broken std::filesystem support in GCC 8
if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(coloring PUBLIC stdc++fs)
  endif()
endif()

install(
  TARGETS coloring
  EXPORT coloring-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_executable(example
    example.cpp
    )

# add_executable(sfd_extract sfd_extract.cpp)
# target_compile_features(sfd_extract PUBLIC cxx_std_17)

target_link_libraries(example PRIVATE coloring)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/coloring-config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/coloring
)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/coloring-config-version.cmake"
  COMPATIBILITY AnyNewerVersion)

install(FILES
  "${PROJECT_BINARY_DIR}/coloring-config.cmake"
  "${PROJECT_BINARY_DIR}/coloring-config-version.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/coloring)

install(
  EXPORT coloring-targets
  NAMESPACE coloring::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/coloring)
