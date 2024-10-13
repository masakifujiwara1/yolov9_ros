from typing import List, Dict

import rospy

from cv_bridge import CvBridge

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints
from torch import cuda

from sensor_msgs.msg import Image, CompressedImage
from yolov9_msgs.msg import Point2D
from yolov9_msgs.msg import BoundingBox2D
from yolov9_msgs.msg import Mask
from yolov9_msgs.msg import KeyPoint2D
from yolov9_msgs.msg import KeyPoint2DArray
from yolov9_msgs.msg import Detection
from yolov9_msgs.msg import DetectionArray

from std_srvs.srv import SetBool


class Yolov9Node:

    def __init__(self):
        rospy.init_node("yolov9_node", anonymous=True)

        self.model = rospy.get_param("~model", "yolov8m.pt")
        self.device = rospy.get_param("~device", "cuda:0")
        self.threshold = rospy.get_param("~threshold", 0.5)
        self.enable = rospy.get_param("~enable", True)

        self.get_logger().info('Yolov9Node created')

        self.cv_bridge = CvBridge()

        self.yolo = YOLO(self.model)
        self.yolo.fuse()

        self._pub = rospy.Publisher("detections", DetectionArray, queue_size=10)
        self._srv = rospy.Service("enable", SetBool, self.enable_cb)

        self._sub = rospy.Subscriber("image_raw", CompressedImage, self.image_cb)

    def enable_cb(self, request):
        self.enable = request.data
        return {"success": True}

    def parse_hypothesis(self, results: Results) -> List[Dict]:

        hypothesis_list = []

        box_data: Boxes
        for box_data in results.boxes:
            hypothesis = {
                "class_id": int(box_data.cls),
                "class_name": self.yolo.names[int(box_data.cls)],
                "score": float(box_data.conf)
            }

            hypothesis_list.append(hypothesis)

        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:

        boxes_list = []

        box_data: Boxes
        for box_data in results.boxes:

            msg = BoundingBox2D()

            box = box_data.xywh[0]
            msg.center.position.x = float(box[0])
            msg.center.position.y = float(box[1])
            msg.size.x = float(box[2])
            msg.size.y = float(box[3])

            boxes_list.append(msg)

        return boxes_list

    def parse_masks(self, results: Results) -> List[Mask]:

        masks_list = []

        def create_point2d(x: float, y: float) -> Point2D:
            p = Point2D()
            p.x = x
            p.y = y
            return p

        mask: Masks
        for mask in results.masks:

            msg = Mask()

            msg.data = [create_point2d(float(ele[0]), float(ele[1]))
                        for ele in mask.xy[0].tolist()]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]

            masks_list.append(msg)

        return masks_list

    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:

        keypoints_list = []

        points: Keypoints
        for points in results.keypoints:

            msg_array = KeyPoint2DArray()

            if points.conf is None:
                continue

            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):

                if conf >= self.threshold:
                    msg = KeyPoint2D()

                    msg.id = kp_id + 1
                    msg.point.x = float(p[0])
                    msg.point.y = float(p[1])
                    msg.score = float(conf)

                    msg_array.data.append(msg)

            keypoints_list.append(msg_array)

        return keypoints_list

    def image_cb(self, msg: CompressedImage) -> None:
        if self.enable:
            cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
            results = self.yolo.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=self.threshold,
                device=self.device
            )
            results: Results = results[0].cpu()

            if results.boxes:
                hypothesis = self.parse_hypothesis(results)
                boxes = self.parse_boxes(results)

            if results.masks:
                masks = self.parse_masks(results)

            if results.keypoints:
                keypoints = self.parse_keypoints(results)

            detections_msg = DetectionArray()

            for i in range(len(results)):
                aux_msg = Detection()

                if results.boxes:
                    aux_msg.class_id = hypothesis[i]["class_id"]
                    aux_msg.class_name = hypothesis[i]["class_name"]
                    aux_msg.score = hypothesis[i]["score"]

                    aux_msg.bbox = boxes[i]

                if results.masks:
                    aux_msg.mask = masks[i]

                if results.keypoints:
                    aux_msg.keypoints = keypoints[i]

                if aux_msg.class_id == 0:
                    detections_msg.detections.append(aux_msg)

            detections_msg.header = msg.header
            self._pub.publish(detections_msg)

            del results
            del cv_image


if __name__ == '__main__':
    node = Yolov9Node()
    rospy.spin()
