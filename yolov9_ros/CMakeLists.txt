cmake_minimum_required(VERSION 3.0.2)
project(yolov9_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/yolov9_node.py
  scripts/debug_node.py
  scripts/tracking_node.py
  # scripts/detect_3d_node.py
  scripts/bev2graph.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY scripts
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
)

