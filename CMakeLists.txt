cmake_minimum_required(VERSION 3.16)
project(pick_ik)

if(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
    add_compile_options(-Werror -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wold-style-cast)
endif()

find_package(catkin REQUIRED
COMPONENTS
  # fmt
  moveit_core
  pluginlib
  # range-v3
  roscpp
  # rsl
  tf2_eigen
  tf2_geometry_msgs
  tf2_kdl
  tl_expected
)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    # fmt
    moveit_core
    pluginlib
    # range-v3
    roscpp
    # rsl
    tf2_eigen
    tf2_geometry_msgs
    tf2_kdl
    tl_expected
  )

include_directories(SYSTEM include
  ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS}
)

add_library(pick_ik_plugin SHARED
  src/fk_moveit.cpp
  src/forward_kinematics.cpp
  src/pick_ik_plugin.cpp
  src/goal.cpp
  src/ik_memetic.cpp
  src/ik_gradient.cpp
  src/robot.cpp
)
target_compile_features(pick_ik_plugin PUBLIC c_std_99 cxx_std_17)
target_include_directories(pick_ik_plugin PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(pick_ik_plugin ${catkin_LIBRARIES})

# pluginlib_export_plugin_description_file(
#   moveit_core
#   pick_ik_kinematics_description.xml
# )

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

install(
  DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(
  TARGETS pick_ik_plugin # pick_ik_parameters
  EXPORT export_pick_ik
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  INCLUDES DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

