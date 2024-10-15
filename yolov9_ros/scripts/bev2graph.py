#!/usr/bin/env python
import rospy
import math
from collections import defaultdict, deque
from sensor_msgs.msg import LaserScan
from yolov9_msgs.msg import DetectionArray
from ptp_msgs.msg import PedestrianArray
from geometry_msgs.msg import PoseArray, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np

class Bev2GraphNode:
    def __init__(self):
        rospy.init_node('bev2graph_node', anonymous=True)
        self.scan_sub = rospy.Subscriber(
            '/scan',
            LaserScan,
            self.callback)
        self.detections_sub = rospy.Subscriber(
            '/yolo/tracking',
            DetectionArray,
            self.callback_yolo)
        self.marker_array_pub = rospy.Publisher('detect_human', MarkerArray, queue_size=10)
        self.pedestrian_array_pub = rospy.Publisher('ped_seq', PedestrianArray, queue_size=10)
        self.dicts = defaultdict(lambda: {'id': 0, 'score': 0, 'theta': 0, 'x': 0, 'y': 0, 'size_x': 0, 'size_y': 0})
        self.scan = LaserScan()
        self.marker_array = MarkerArray()

        # process frame
        self.frame = 0
        self.data_array = None
        self.is_fst_flag = True
        self.curr_frames = deque(maxlen=8)
        rospy.Timer(rospy.Duration(0.4), self.process_frames)

    def callback_yolo(self, data):
        self.dicts.clear()
        self.marker_array = MarkerArray()
        for idx, i in enumerate(data.detections):
            if i.class_id == 0:
                self.dicts[idx]['id'] = i.id
                self.dicts[idx]['score'] = i.score
                self.dicts[idx]['theta'] = i.bbox.center.theta
                self.dicts[idx]['size_x'] = i.bbox.size.x
                self.dicts[idx]['size_y'] = i.bbox.size.y

    def calc_pose(self):
        for key, value in self.dicts.items():
            angle = value['theta']
            id_ = value['id']
            score = value['score']
            size_x = value['size_x']
            size_y = value['size_y']

            if angle < self.scan.angle_min or angle > self.scan.angle_max:
                rospy.logwarn(f'指定した角度が範囲外です. angle={angle}, min={self.scan.angle_min}')
                return

            index = int((angle - self.scan.angle_min) / self.scan.angle_increment)

            if 0 <= index < len(self.scan.ranges):
                min_dist = float('inf')
                for offset in range(-7, 7):
                    distance = self.scan.ranges[index + offset]
                    min_dist = min(distance, min_dist)
                
                distance = min_dist
                x, y = self.calc_xy(angle, distance)

                self.dicts[key]['x'] = x
                self.dicts[key]['y'] = y

                if distance != float('inf'):
                    self.add_marker(id_, x, y)
            else:
                rospy.logwarn('計算されたインデックスが範囲外です')

        self.publish_marker_array()

    def callback(self, scan):
        self.scan = scan

    def calc_xy(self, angle, distance):
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        return x, y

    def is_valid_index(self, i, length):
        return 0 <= i < length

    def add_marker(self, id_, x, y):
        point = Marker()
        point.header.frame_id = 'base_scan'
        point.ns = "point"
        point.id = int(id_)
        point.type = Marker.SPHERE
        point.action = Marker.ADD
        point.pose.position.x = x
        point.pose.position.y = y
        point.pose.position.z = 0.0
        point.pose.orientation.x = 0
        point.pose.orientation.y = 0
        point.pose.orientation.z = 0
        point.pose.orientation.w = 1
        point.scale.x = point.scale.y = point.scale.z = 0.5
        point.color.r = 0.0
        point.color.g = 1.0
        point.color.b = 0.0
        point.color.a = 1.0
        point.lifetime = rospy.Duration(0.4)
        self.marker_array.markers.append(point)
    
    def publish_marker_array(self):
        self.marker_array_pub.publish(self.marker_array)

    def process_frames(self, event):
        self.curr_frames.append(self.frame)
        self.calc_pose()
        for key, value in self.dicts.items():
            data = np.array([self.frame, self.dicts[key]['id'], self.dicts[key]['x'], self.dicts[key]['y']], dtype=np.float32)

            if self.is_fst_flag:
                self.data_array = data
                self.is_fst_flag = False
            else:
                self.data_array = np.vstack((self.data_array, data))

        if len(self.curr_frames) == 8:
            self.data_array = self.data_array[self.data_array[:, 0].astype(int) >= self.curr_frames[0]]

            msg = PedestrianArray()
            msg.data = self.data_array.flatten().tolist()
            msg.shape = self.data_array.shape
            msg.dtype = str(self.data_array.dtype)

            self.pedestrian_array_pub.publish(msg)

        self.frame += 24

def main():
    bev2graph_node = Bev2GraphNode()
    rospy.spin()

if __name__ == '__main__':
    main()
