
import torch
from transformers import MobileNetV2Config, MobileNetV2Model
from transformers.models.mobilenet_v2.modeling_mobilenet_v2 import MobileNetV2InvertedResidual

def debug_mobilenet():
    config = MobileNetV2Config(depth_multiplier=0.25)
    model = MobileNetV2Model(config)
    model.eval()

    pixel_values = torch.randn(1, 3, 224, 224)
    outputs = model(pixel_values, output_hidden_states=True, return_dict=True)
    
    hidden_states = outputs.hidden_states
    print(f"Number of hidden states: {len(hidden_states)}")
    
    for i, hs in enumerate(hidden_states):
        print(f"Hidden state {i} shape: {hs.shape}")

    # Check _can_record_outputs
    print(f"_can_record_outputs: {model._can_record_outputs}")
    
    # Check if stem is captured?
    # model.conv_stem output shape should be [1, 8, 112, 112]
    
if __name__ == "__main__":
    debug_mobilenet()
