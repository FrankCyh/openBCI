from model import ConvNet
import os
import torch

model = ConvNet()

pretrained_model_name = 'final_model_epoch_50000_bs_128-newtrain-4people-conf.pth'
pretrained_model_path = os.path.join('pretrained_models', 'Nov25_balanced_50000', pretrained_model_name)
model.load_state_dict(torch.load(pretrained_model_path))
model.eval()

dummy_input = torch.randn(1, 1, 125, 5)
onnx_path = 'saved_models/test_opset_version_16.onnx'
torch.onnx.export(model, dummy_input, onnx_path, opset_version=16)