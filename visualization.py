"""
Visualization and results functions
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(train_losses, val_losses, train_dice, val_dice, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Train', marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='Validation', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(epochs, train_dice, 'b-', label='Train', marker='o')
    ax2.plot(epochs, val_dice, 'r-', label='Validation', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Training and Validation Dice')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_predictions(model, loader, device, num_samples, save_path):
    model.eval()
    imgs_list, masks_list, preds_list = [], [], []
    
    with torch.no_grad():
        for imgs, masks in loader:
            outputs = model(imgs.to(device))
            preds = torch.argmax(outputs, dim=1)
            imgs_list.append(imgs.cpu())
            masks_list.append(masks.cpu())
            preds_list.append(preds.cpu())
            if len(imgs_list) >= num_samples:
                break
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        axes[i, 0].imshow(imgs_list[i][0, 0], cmap='gray')
        axes[i, 0].set_title('CT')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(masks_list[i][0], cmap='jet')
        axes[i, 1].set_title(f'Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(preds_list[i][0], cmap='jet')
        axes[i, 2].set_title(f'Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results(results, model_info, save_path):
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("VERTEBRAE SEGMENTATION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("MODEL\n")
        f.write("-"*70 + "\n")
        f.write(f"Architecture: U-Net\n")
        f.write(f"Parameters: {model_info['params']:,}\n")
        f.write(f"Training time: {model_info['time']:.1f} min\n")
        f.write(f"Epochs: {model_info['epochs']}\n\n")
        
        f.write("DATASET\n")
        f.write("-"*70 + "\n")
        f.write(f"Training scans: {model_info['train_scans']}\n")
        f.write(f"Validation scans: {model_info['val_scans']}\n")
        f.write(f"Test scans: {model_info['test_scans']}\n\n")
        
        for name, res in results.items():
            f.write(f"{name.upper()} SET\n")
            f.write("-"*70 + "\n")
            f.write(f"Dice: {res['dice_mean']:.4f} ± {res['dice_std']:.4f}\n")
            f.write(f"IoU: {res['iou_mean']:.4f}\n")
            f.write(f"Accuracy: {res['acc_mean']*100:.2f}%\n\n")
