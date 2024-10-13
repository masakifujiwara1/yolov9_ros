#!/usr/bin/env python
import numpy as np
import math

import rospy
from cv_bridge import CvBridge

from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml
from ultralytics.engine.results import Boxes

from sensor_msgs.msg import Image, CompressedImage
from yolov9_msgs.msg import Detection
from yolov9_msgs.msg import DetectionArray
from std_srvs.srv import SetBool
import message_filters


class TrackingNode:

    def __init__(self) -> None:
        rospy.init_node("tracking_node")

        self.cv_bridge = CvBridge()

        tracker_name = rospy.get_param("~tracker", "bytetrack.yaml")

        self.tracker = self.create_tracker(tracker_name)
        self._pub = rospy.Publisher("tracking", DetectionArray, queue_size=10)

        # subs
        self.image_sub = message_filters.Subscriber(
            "/image_raw", CompressedImage)
        self.detections_sub = message_filters.Subscriber(
            "/detections", DetectionArray)

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.detections_sub], 10, 0.5)
        self._synchronizer.registerCallback(self.detections_cb)

    def create_tracker(self, tracker_yaml: str) -> BaseTrack:

        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        check_requirements("lap")  # for linear_assignment

        tracker = check_yaml(tracker_yaml)
        cfg = IterableSimpleNamespace(**yaml_load(tracker))

        assert cfg.tracker_type in ["bytetrack", "botsort"], \
            f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=1)
        return tracker

    def calc_theta(self, w_img_, x_center_):
        fov_horizonal = math.pi / 2  # 90deg
        theta_ = -((x_center_ - (w_img_ / 2)) * fov_horizonal) / w_img_
        theta_ = math.atan2(math.sin(theta_), math.cos(theta_))
        return theta_

    def detections_cb(self, img_msg: CompressedImage, detections_msg: DetectionArray) -> None:

        tracked_detections_msg = DetectionArray()
        tracked_detections_msg.header = img_msg.header

        # convert image
        cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(img_msg)

        # parse detections
        detection_list = []
        detection: Detection
        for detection in detections_msg.detections:

            detection_list.append(
                [
                    detection.bbox.center.position.x - detection.bbox.size.x / 2,
                    detection.bbox.center.position.y - detection.bbox.size.y / 2,
                    detection.bbox.center.position.x + detection.bbox.size.x / 2,
                    detection.bbox.center.position.y + detection.bbox.size.y / 2,
                    detection.score,
                    detection.class_id
                ]
            )

        img_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

        # tracking
        if len(detection_list) > 0:

            det = Boxes(
                np.array(detection_list),
                (img_msg.height, img_msg.width)  # 720, 1280 fov H: 90 V: 59
            )

            tracks = self.tracker.update(det, cv_image)

            if len(tracks) > 0:

                for t in tracks:

                    tracked_box = Boxes(
                        t[:-1], (img_msg.height, img_msg.width))

                    tracked_detection: Detection = detections_msg.detections[int(
                        t[-1])]

                    # get boxes values
                    box = tracked_box.xywh[0]
                    tracked_detection.bbox.center.position.x = float(box[0])
                    tracked_detection.bbox.center.position.y = float(box[1])

                    tracked_detection.bbox.center.theta = self.calc_theta(img_msg.width, float(box[0]))

                    tracked_detection.bbox.size.x = float(box[2])
                    tracked_detection.bbox.size.y = float(box[3])

                    # get track id
                    track_id = ""
                    if tracked_box.is_track:
                        track_id = str(int(tracked_box.id))
                    tracked_detection.id = track_id

                    # append msg
                    tracked_detections_msg.detections.append(tracked_detection)

        # publish detections
        self._pub.publish(tracked_detections_msg)


if __name__ == '__main__':
    node = TrackingNode()
    rospy.spin()
