import cv2
import datetime
import onnxruntime
import numpy as np
from insightface.model_zoo.scrfd import SCRFD
from insightface.utils.face_align import norm_crop
from insightface.model_zoo.arcface_onnx import ArcFaceONNX


scrfd_path = "scrfd_10g_gnkps.onnx"
scrfd_session = onnxruntime.InferenceSession(scrfd_path, providers=['CUDAExecutionProvider'])
scrfd = SCRFD(scrfd_path, scrfd_session)

arcface_path = "resnet18_110.onnx"

arcface_session = onnxruntime.InferenceSession(arcface_path, providers=['CUDAExecutionProvider'])
arcface = ArcFaceONNX(arcface_path, arcface_session)

def get_feature(img):
    """
    计算图片512位特征值
    :param img: 图片
    :return 特征值
    """
    # 去除运动模糊
    # kernels = np.ones([5, 5], np.float32) / 25
    # img = cv2.filter2D(img, -1, kernel=kernels)
    # 变为灰色系
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化
    img = cv2.equalizeHist(img)
    # 设置大小
    output = arcface.get_feat(img)
    return np.squeeze(output)

def get_pic_feature(path):
    frame = cv2.imread(path)
    bboxes, kpss = scrfd.detect(frame, input_size=(640, 640))
    bbox = bboxes[0]
    x1, y1, x2, y2, score = bbox.astype(np.int)
    w, h = x2-x1, y2-y1
    # face = Face(bbox=bbox, kps=kpss, det_score=0)
    start_x =  int(max(x1 - w/2, 0))
    end_x =  int(min(720, x1 + 1.5*w))
    start_y =  int(max(y1 - h/2, 0))
    end_y =  int(min(1280, y1 + 1.5*h))
    cur_frame = frame[start_y:end_y, start_x:end_x]
    # 这一步必须有，否则图像无法显示
    kpss = [[key_pos[0]-start_x, key_pos[1]-start_y]for key_pos in kpss[0]]
    face_roi = norm_crop(cur_frame, np.array(kpss))
    # cv2.imshow('frame', face_roi)
    # # 这一步必须有，否则图像无法显示
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     pass
    # 这一步必须有，否则图像无法显示
    bboxes_second, kpss_second = scrfd.detect(face_roi, input_size=(640, 640))
    print("sss", bboxes_second, kpss_second)
    x1, y1, x2, y2, _ = bboxes_second[0].astype(np.int)
    x1, y1 = max(0, x1), max(0, y1)
    # 截取人脸图片
    face_img = face_roi[y1:y2, x1:x2]
    #人脸面积
    face_area = (y2-y1)*(x2-x1)
    feature  = get_feature(face_img)
    return feature


stb = get_pic_feature("scx.jpg")
    

video = cv2.VideoCapture(0)
while True:
    flag, frame = video.read()
    if not flag:
        break
    ta = datetime.datetime.now()
    bboxes, kpsses = scrfd.detect(frame, input_size=(640, 640))
    tb = datetime.datetime.now()
    print('all cost:', (tb - ta).total_seconds() * 1000)
    try:
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x11, y11, x12, y12, score = bbox.astype(np.int)
            w, h = x12-x11, y12-y11
            face_area =w*h
            # face = Face(bbox=bbox, kps=kpss, det_score=0)
            start_x =  int(max(x11 - w/2, 0))
            end_x =  int(min(1920, x11 + 1.5*w))
            start_y =  int(max(y11 - h/2, 0))
            end_y =  int(min(1080, y11 + 1.5*h))
            cur_frame = frame[start_y:end_y, start_x:end_x]
            kpss = [[key_pos[0]-start_x, key_pos[1]-start_y]for key_pos in kpsses[i]]
            face_roi = norm_crop(cur_frame, np.array(kpss))
            bboxes_second, kpss_second = scrfd.detect(face_roi, input_size=(640, 640))
            x1, y1, x2, y2, _ = bboxes_second[0].astype(np.int)
            x1, y1 = max(0, x1), max(0, y1)
            # 截取人脸图片
            face_img = face_roi[y1:y2, x1:x2]
            #人脸面积
            # face_area = (y2-y1)*(x2-x1)
            feature  = get_feature(face_img)
            score = arcface.compute_sim(stb, feature)
            cv2.rectangle(frame, (x11, y11), (x12, y12), (255, 0, 0), 2)
            cv2.putText(frame, str(score)[:4], (x11, y11), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(frame, str(face_area), (x11, y11+h), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            if kpsses is not None:
                kps = kpsses[i]
                for kp in kps:
                    kp = kp.astype(np.int)
                    cv2.circle(frame, tuple(kp), 1, (0, 0, 255), 2)
    except Exception:
        pass
    # frame = cv2.flip(
    #         frame,
    #         1  # 1：水平镜像，-1：垂直镜像
    #     )
    cv2.imshow('frame', frame)
    # 这一步必须有，否则图像无法显示
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 当一切完成时，释放捕获
video.release()
cv2.destroyAllWindows()
