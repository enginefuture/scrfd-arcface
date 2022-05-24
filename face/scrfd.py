import onnxruntime
import cv2
from insightface.model_zoo.scrfd import SCRFD

scrfd_path = "/vlogpy/resources/models/scrfd_10g_gnkps.onnx"

scrfd_session = onnxruntime.InferenceSession(scrfd_path, providers=['CUDAExecutionProvider'])
scrfd = SCRFD(scrfd_path, scrfd_session)
