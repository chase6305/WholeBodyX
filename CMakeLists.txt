cmake_minimum_required(VERSION 3.20)
project(WholeBodyX VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)
option(WBX_BUILD_PYTHON "Build optional Python extension" OFF)
option(WBX_BUILD_EXAMPLES "Build C++ examples" ON)

find_package(Eigen3 3.4 CONFIG REQUIRED)
find_package(osqp CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_library(wholebodyx cpp/control.cpp cpp/osqp_solver.cpp cpp/joint_order.cpp cpp/runtime.cpp cpp/reference.cpp cpp/sparse_mpc.cpp)
add_library(wholebodyx::wholebodyx ALIAS wholebodyx)
target_compile_features(wholebodyx PUBLIC cxx_std_17)
target_include_directories(wholebodyx PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(wholebodyx PUBLIC Eigen3::Eigen PRIVATE osqp::osqp Threads::Threads)
set_target_properties(wholebodyx PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(WBX_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(wholebodyx_native bindings/module.cpp)
  target_link_libraries(wholebodyx_native PRIVATE wholebodyx)
  install(TARGETS wholebodyx_native LIBRARY DESTINATION . COMPONENT python)
endif()

if(WBX_BUILD_EXAMPLES)
  foreach(example joint_control task_weight_tradeoff joint_limit_handling mpc_retargeting joint_order_adapter)
    add_executable(wbx_${example} examples/cpp/${example}.cpp)
    target_link_libraries(wbx_${example} PRIVATE wholebodyx)
    if(BUILD_TESTING)
      add_test(NAME example_${example} COMMAND wbx_${example})
    endif()
  endforeach()
endif()
if(BUILD_TESTING)
  foreach(test control joint_order runtime reference sparse_mpc fixed_mpc concurrency velocity_bounds)
    add_executable(wbx_${test}_tests tests/cpp/${test}_tests.cpp)
    target_link_libraries(wbx_${test}_tests PRIVATE wholebodyx)
    add_test(NAME ${test} COMMAND wbx_${test}_tests)
  endforeach()
endif()

install(TARGETS wholebodyx EXPORT WholeBodyXTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT development
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT development
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT development)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT development)
install(EXPORT WholeBodyXTargets NAMESPACE wholebodyx::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/WholeBodyX COMPONENT development)
configure_package_config_file(cmake/WholeBodyXConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/WholeBodyXConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/WholeBodyX)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/WholeBodyXConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/WholeBodyXConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/WholeBodyXConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/WholeBodyX COMPONENT development)
