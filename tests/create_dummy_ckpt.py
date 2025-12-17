
import torch
import os
from src.models.build_model import build_model

def main():
    os.makedirs('outputs/checkpoints', exist_ok=True)
    model = build_model('resnet18', num_classes=10)
    torch.save(model.state_dict(), 'outputs/checkpoints/dummy.pth')
    print("Saved dummy.pth")

if __name__ == '__main__':
    main()
