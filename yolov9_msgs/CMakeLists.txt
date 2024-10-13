cmake_minimum_required(VERSION 3.0.2)
project(yolov9_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
  Point2D.msg
  Vector2.msg
  Pose2D.msg
  BoundingBox2D.msg
  BoundingBox3D.msg
  Mask.msg
  KeyPoint2D.msg
  KeyPoint2DArray.msg
  KeyPoint3D.msg
  KeyPoint3DArray.msg
  Detection.msg
  DetectionArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs
)
