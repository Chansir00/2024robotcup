import onnx

# 加载模型
model = onnx.load("./pose_estimation_mediapipe_2023mar.onnx")

# 打印模型的graph
print(onnx.helper.printable_graph(model.graph))