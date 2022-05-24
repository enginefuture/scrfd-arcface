import onnxruntime
import cv2
import numpy as np
from insightface.model_zoo.arcface_onnx import ArcFaceONNX

arcface_path = "/vlogpy/resources/models/resnet18_110.onnx"

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
