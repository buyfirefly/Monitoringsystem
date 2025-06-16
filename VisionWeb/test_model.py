import sys
sys.path.append('e:/model/multimodal-emotion-recognition-system-master')

from multimodal_emotion.fusion_model import MultiModalEmotionModel
import torch

try:
    print("Creating model...")
    model = MultiModalEmotionModel(
        visual_dim=256,
        audio_dim=256,
        hidden_dim=512,
        num_classes=7
    )
    print("Model created successfully")
    
    print("Loading checkpoint...")
    checkpoint = torch.load('e:/model/best_model.pth', map_location='cpu', weights_only=False)
    print("Checkpoint loaded successfully")
    
    print("Loading model state dict...")
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()