#!/usr/bin/env python
import rospy
import math
from collections import defaultdict, deque
from sensor_msgs.msg import LaserScan
from yolov9_msgs.msg import DetectionArray
from ptp_msgs.msg import PedestrianArray
from geometry_msgs.msg import PoseArray, Quaternion, Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from tf_pose import *

class Bev2GraphNode:
    def __init__(self):
        rospy.init_node('bev2graph_node', anonymous=True)
        self.scan_sub = rospy.Subscriber(
            '/rfans/surestar_scan',
            LaserScan,
            self.callback)
        self.detections_sub = rospy.Subscriber(
            '/yolo/tracking',
            DetectionArray,
            self.callback_yolo)
        self.marker_array_pub = rospy.Publisher('detect_human', MarkerArray, queue_size=10)
        self.pedestrian_array_pub = rospy.Publisher('ped_seq', PedestrianArray, queue_size=10)
        self.dicts = defaultdict(lambda: {'id': 0, 'score': 0, 'theta': 0, 'x': 0, 'y': 0, 'size_x': 0, 'size_y': 0, 'distance': 0})
        self.scan = LaserScan()
        self.marker_array = MarkerArray()
        self.detection_array = DetectionArray()

        # process frame
        self.frame = 0
        self.data_array = None
        self.is_fst_flag = True
        self.curr_frames = deque(maxlen=8)
        rospy.Timer(rospy.Duration(0.4), self.process_frames)
        rospy.Timer(rospy.Duration(0.1), self.calc_pose)

        # viz setting
        self.color_palette = [
        [255, 0, 0],      # 赤
        [0, 255, 0],      # 緑
        [0, 0, 255],      # 青
        [255, 255, 0],    # 黄
        [255, 0, 255],    # マゼンタ
        [0, 255, 255],    # シアン
        [128, 0, 0],      # 暗赤
        [0, 128, 0],      # 暗緑
        [0, 0, 128],      # 暗青
        [128, 128, 0],    # オリーブ
        [128, 0, 128],    # 紫
        [0, 128, 128],    # ティール
        [192, 192, 192],  # 銀
        [128, 128, 128],  # グレー
        [255, 165, 0],    # オレンジ
        [75, 0, 130]      # インディゴ
        ]
        
        init_tf()

    def callback_yolo(self, data):
        # self.dicts.clear()
        self.marker_array = MarkerArray()
        self.detection_array.detections = data.detections
        # for idx, i in enumerate(data.detections):
        #     if i.class_id == 0:
        #         self.dicts[idx]['id'] = i.id
        #         self.dicts[idx]['score'] = i.score
        #         self.dicts[idx]['theta'] = i.bbox.center.theta
        #         self.dicts[idx]['size_x'] = i.bbox.size.x
        #         self.dicts[idx]['size_y'] = i.bbox.size.y

    def calc_pose(self, event):
        self.dicts.clear()

        for idx, i in enumerate(self.detection_array.detections):
            if i.class_id == 0:
                self.dicts[idx]['id'] = i.id
                self.dicts[idx]['score'] = i.score
                self.dicts[idx]['theta'] = i.bbox.center.theta
                self.dicts[idx]['size_x'] = i.bbox.size.x
                self.dicts[idx]['size_y'] = i.bbox.size.y

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

                # if angle > 0.2:
                #     for offset in range(-5, 15):
                #         distance = self.scan.ranges[index + offset]
                #         min_dist = min(distance, min_dist)
                # elif angle < -0.2:
                #     for offset in range(-15, 5):
                #         distance = self.scan.ranges[index + offset]
                #         min_dist = min(distance, min_dist)
                # else:
                for offset in range(-10, 10):
                    distance = self.scan.ranges[index + offset]
                    min_dist = min(distance, min_dist)
                
                distance = min_dist
                x, y = self.calc_xy(angle, distance)

                x, y, _ = transform_pose(x, y, 0.0)

                self.dicts[key]['x'] = x
                self.dicts[key]['y'] = y
                self.dicts[key]['distance'] = distance

                # if distance != float('inf'):
                #     self.add_marker(id_, x, y)
            else:
                rospy.logwarn('計算されたインデックスが範囲外です')

        self.add_tracking_marker()
        self.publish_marker_array()

    def callback(self, scan):
        self.scan = scan

    def calc_xy(self, angle, distance):
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        return x, y

    def is_valid_index(self, i, length):
        return 0 <= i < length

    def add_marker(self, id_, x_, y_):
        point = Marker()
        point.header.frame_id = 'map'
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
    
    def add_tracking_marker(self):
        boxes = Marker()
        boxes.header.frame_id = 'odom'
        boxes.action = Marker.ADD
        boxes.lifetime = rospy.Duration(1.0)
        boxes.ns = "boxes"
        boxes.type = Marker.CUBE_LIST
        boxes.scale.x = 0.5
        boxes.scale.y = 0.5
        boxes.scale.z = 1.2
        # boxes.pose.position.x = x_
        # boxes.pose.position.y = y_
        boxes.pose.position.z = 0.0
        boxes.pose.orientation.w = 1.0

        for key, value in self.dicts.items():
            color = self.color_palette[int(value['id']) % len(self.color_palette)]

            if value['distance'] == float('inf'):
                continue

            if abs(self.dicts[key]['theta']) >= 2.62:
                continue

            rgba = ColorRGBA()
            rgba.r = color[2] / 255.0
            rgba.g = color[1] / 255.0
            rgba.b = color[0] / 255.0
            rgba.a = 0.6
            boxes.colors.append(rgba)

            point = Point()
            point.x = float(value['x'])
            point.y = float(value['y'])
            point.z = 0.0
            boxes.points.append(point)

            text = Marker()
            text.header = boxes.header
            text.action = Marker.ADD
            text.lifetime = rospy.Duration(1.0)
            text.ns = f"text{value['id']}"
            text.type = Marker.TEXT_VIEW_FACING
            text.scale.z = 0.5
            text.pose.position = point
            text.pose.position.z += 0.7
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"id:{value['id']}"
            
            self.marker_array.markers.append(text)
        self.marker_array.markers.append(boxes)
    
    def publish_marker_array(self):
        self.marker_array_pub.publish(self.marker_array)

    def process_frames(self, event):
        self.curr_frames.append(self.frame)
        # self.calc_pose()
        try:
            for key, value in self.dicts.items():

                if abs(self.dicts[key]['theta']) >= 2.62:
                    continue
                
                data = np.array([self.frame, self.dicts[key]['id'], self.dicts[key]['x'], self.dicts[key]['y']], dtype=np.float32)

                if self.is_fst_flag:
                    self.data_array = data
                    self.is_fst_flag = False
                else:
                    self.data_array = np.vstack((self.data_array, data))
        except:
            rospy.loginfo("error")

        if len(self.curr_frames) == 8:
            try:
                self.data_array = self.data_array[self.data_array[:, 0].astype(int) >= self.curr_frames[0]]
            except:
                rospy.loginfo("error")
                return

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
