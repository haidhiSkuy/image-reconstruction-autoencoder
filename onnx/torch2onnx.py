import torch
from model import get_model 
from load_model import load_params 

model = get_model()
model = load_params(model, "epoch=9-loss=0.00.ckpt")

model.eval()
dummy_input = torch.randn(1, 3, 256, 256)

input_names = ["input"]
output_names = ["output"]

torch.onnx.export(
    model,
    dummy_input,
    "autoencoder.onnx",
    verbose=False,
    input_names=input_names,
    output_names=output_names,
    export_params=True,
)