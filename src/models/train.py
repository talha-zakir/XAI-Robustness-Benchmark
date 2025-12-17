
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, trainloader, testloader, epochs=10, lr=0.001, device='cuda', save_path='checkpoint.pth'):
    """
    Standard training loop.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
            
        # Validation
        acc = evaluate_model(model, testloader, device)
        print(f"Epoch {epoch+1} Test Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            print(f"Saving best model with acc {best_acc:.2f}%")
            torch.save(model.state_dict(), save_path)
            
    return best_acc

def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100. * correct / total
