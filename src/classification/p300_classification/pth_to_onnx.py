from model import ConvNet
import os
import torch

model = ConvNet()

pretrained_model_name = 'model_epoch_1000_bs_128-newtrain-4people.pth'
pretrained_model_path = os.path.join('pretrained_models', pretrained_model_name)
model.load_state_dict(torch.load(pretrained_model_path))
model.eval()

dummy_input = torch.randn(1, 1, 125, 5)
onnx_path = 'saved_models/4people.onnx'
torch.onnx.export(model, dummy_input, onnx_path)