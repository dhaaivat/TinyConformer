import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
                device='cuda', num_epochs=20, save_path='best_conformer_model.pth',
                patience=5, delta=0.0):
    
    model.to(device)
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\n[INFO] Epoch {epoch+1}/{num_epochs}")
        model.train()  # Enables SpecAugment internally
        total_loss, correct, total = 0.0, 0, 0

        loop = tqdm(train_loader, desc="Training")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)  # SpecAugment applied inside if in train mode
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        val_acc = evaluate_model(model, val_loader, device)

        if val_acc > best_val_acc + delta:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            torch.save(best_model_state, save_path)
            print(f"[INFO] New best model saved with val acc: {val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"[INFO] No improvement. Patience left: {patience - epochs_no_improve}")
            if epochs_no_improve >= patience:
                print(f"[Early Stopping] No improvement for {patience} epochs.")
                break

        if scheduler:
            scheduler.step()

    print(f"\n Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    return best_val_acc


def evaluate_model(model, loader, device='cuda'):
    model.eval()  # Disables SpecAugment
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = 100. * correct / total
    print(f" Validation Accuracy: {acc:.2f}%")
    return acc
