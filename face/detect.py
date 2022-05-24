import cv2
import numpy as np



from .euler import euler
from .scrfd import scrfd
from insightface.app.common import Face
from insightface.utils.face_align import norm_crop
from config import config


class FacialRecognitionTechnology:
    """人脸识别类"""

    def detection_face(self, frame, bbox, kpss):
        """对图片重新进行截取"""
        x1, y1, x2, y2 = bbox
        w, h = x2-x1, y2-y1
        face = Face(bbox=bbox, kps=kpss, det_score=0)
        # 进行欧拉角判定   
        euler.get(frame, face)
        pitch, roll, yaw = face.pose
        if abs(int(pitch)) > config.euler_angle.get("pitch") or abs(int(roll)) > config.euler_angle.get("roll") or abs(int(yaw)) > config.euler_angle.get("yaw"):
            return [], -1
        start_x =  int(max(x1 - w/2, 0))
        end_x =  int(min(1920, x1 + 1.5*w))
        start_y =  int(max(y1 - h/2, 0))
        end_y =  int(min(1080, y1 + 1.5*h))
        cur_frame = frame[start_y:end_y, start_x:end_x]
        kpss = [[key_pos[0]-start_x, key_pos[1]-start_y]for key_pos in kpss]
        face_roi = norm_crop(cur_frame, np.array(kpss))
        bboxes, kpss = scrfd.detect(face_roi, input_size=(640, 640))
        # 如果没有识别到人脸
        if len(bboxes) == 0:
            return [], -1
        x1, y1, x2, y2, _ = bboxes[0].astype(np.int)
        x1, y1 = max(0, x1), max(0, y1)
        # 截取人脸图片
        face_img = face_roi[y1:y2, x1:x2]
        # 判断人脸是否清晰
        # gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        # if fm < 20:
        #     return [], -1
        # 计算人脸面积
        face_area = (y2-y1)*(x2-x1)
        return face_img, face_area


face_reg = FacialRecognitionTechnology()
