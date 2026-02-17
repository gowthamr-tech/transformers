from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("google/mobilenet_v2_1.0_224")

dummy_input = torch.randn(1, 3, 224, 224)
outputs = model(dummy_input)

print("Loaded successfully")
print(type(outputs))
