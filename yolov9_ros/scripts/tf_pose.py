#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped

tf_buffer = None
tf_listener = None

def init_tf():
    global tf_buffer, tf_listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

def transform_pose(x_, y_, z_):

    if tf_buffer is None or tf_listener is None:
        raise RuntimeError("tf_buffer or tf_listener is not initialized. Call init_tf() first.")

    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = "base_link"
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x_
    pose_stamped.pose.position.y = y_
    pose_stamped.pose.position.z = z_
    pose_stamped.pose.orientation.w = 1.0

    try:
        transform = tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
        transformed_pose = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
        return transformed_pose.pose.position.x, transformed_pose.pose.position.y, transformed_pose.pose.position.z

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.loginfo(f"Transform failed: {e}")
        return None

if __name__ == '__main__':
    transform_pose(0.0, 0.0, 0.0)
