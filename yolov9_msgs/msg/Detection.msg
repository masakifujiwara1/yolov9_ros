# defines a YOLOv9 detection result

# class probability
int32 class_id
string class_name
float64 score

# ID for tracking
string id

# 2D bounding box surrounding the object in pixels
yolov9_msgs/BoundingBox2D bbox

# 3D bounding box surrounding the object in meters
yolov9_msgs/BoundingBox3D bbox3d

# segmentation mask of the detected object
# it is only the boundary of the segmented object
yolov9_msgs/Mask mask

# keypoints for human pose estimation
yolov9_msgs/KeyPoint2DArray keypoints

# keypoints for human pose estimation
yolov9_msgs/KeyPoint3DArray keypoints3d