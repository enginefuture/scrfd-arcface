import onnxruntime
from insightface.model_zoo.landmark import Landmark

euler_path = "/vlogpy/resources/models/1k3d68.onnx"

euler_session = onnxruntime.InferenceSession(euler_path, providers=['CUDAExecutionProvider'])
euler = Landmark(euler_path, euler_session)
