import torch
import torch.nn
import onnx
 
model = torch.load(r'X:\robot\2024\nwsztsg\best.pt')
model.eval()
 
input_names = ['input']
output_names = ['output']
 
x = torch.randn(1,3,32,32,requires_grad=True)
 
torch.onnx.export(model, x, 'best.onnx', input_names=input_names, output_names=output_names, verbose='True')
#python export.py --weights runs/train/exp7/best.pt --include onnx --opset 12 --dynamic