
import argparse
import os
import torch
from src.datasets.cifar10 import load_cifar10
from src.models.build_model import build_model
from src.models.train import train_model

def main():
    parser = argparse.ArgumentParser(description='Train Baseline Model')
    parser.add_argument('--data_root', type=str, default='./data', help='path to dataset')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs') # Reduced default for quick testing
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--output_dir', type=str, default='./outputs/checkpoints', help='where to save model')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load Data
    trainloader, testloader, _ = load_cifar10(args.data_root, batch_size=args.batch_size)
    
    # Build Model
    model = build_model('resnet18', num_classes=10)
    
    # Train
    save_path = os.path.join(args.output_dir, 'resnet18_cifar10_baseline.pth')
    train_model(model, trainloader, testloader, epochs=args.epochs, lr=args.lr, device=device, save_path=save_path)
    
    print("Training Complete.")

if __name__ == '__main__':
    main()
