"""
Comprehensive Model Evaluation with Additional Visualizations
Generates: Confusion Matrix, ROC Curve, Precision-Recall, F1 Score, etc.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report, f1_score, precision_score, recall_score
)
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from data_loader import load_dataset
from model import UNet
from dataset import VerseDataset


CONFIG = {
    'val_path': 'Dataset/dataset-verse19validation',
    'test_path': 'Dataset/dataset-verse19test',
    'model_path': 'results/best_model.pth',
    'results_dir': 'results/evaluation',
    'batch_size': 4,
    'device': torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
}


def collect_predictions(model, loader, device):
    """Collect all predictions and ground truth"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Collecting predictions'):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            
            # Get probabilities (softmax)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of vertebrae class
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
    
    return np.array(all_preds), np.array(all_targets), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Background', 'Vertebrae'],
                yticklabels=['Background', 'Vertebrae'],
                cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.75, f'({percentage:.2f}%)', 
                   ha='center', va='center', fontsize=10, color='gray')
    
    # Add metrics as text
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f'Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}'
    ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return cm


def plot_roc_curve(y_true, y_probs, save_path):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
           label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return roc_auc


def plot_precision_recall_curve(y_true, y_probs, save_path):
    """Plot Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(recall, precision, color='blue', lw=2,
           label=f'PR curve (AUC = {pr_auc:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return pr_auc


def plot_metrics_comparison(val_metrics, test_metrics, save_path):
    """Plot comparison of metrics between validation and test sets"""
    metrics = ['Dice', 'IoU', 'Precision', 'Recall', 'F1-Score', 'Accuracy']
    val_values = [val_metrics[k] for k in metrics]
    test_values = [test_metrics[k] for k in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, val_values, width, label='Validation', color='steelblue')
    bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='darkorange')
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics: Validation vs Test', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate all metrics"""
    # Basic metrics
    dice_scores = []
    iou_scores = []
    
    # Calculate per-sample metrics (for std)
    unique_samples = len(y_true) // (256 * 256)  # Approximate number of images
    samples_per_image = 256 * 256
    
    for i in range(0, len(y_true), samples_per_image):
        pred_sample = y_pred[i:i+samples_per_image]
        true_sample = y_true[i:i+samples_per_image]
        
        inter = np.sum(pred_sample * true_sample)
        union = np.sum(pred_sample) + np.sum(true_sample)
        
        if union > 0:
            dice = (2 * inter) / union
            iou = inter / (union - inter) if (union - inter) > 0 else 0
            dice_scores.append(dice)
            iou_scores.append(iou)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = np.mean(y_pred == y_true)
    
    # ROC and PR AUC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(rec_curve, prec_curve)
    
    return {
        'Dice': np.mean(dice_scores),
        'Dice_std': np.std(dice_scores),
        'IoU': np.mean(iou_scores),
        'IoU_std': np.std(iou_scores),
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Accuracy': accuracy,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc
    }


def save_metrics_table(val_metrics, test_metrics, save_path):
    """Save detailed metrics table"""
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE EVALUATION METRICS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Metric':<20} {'Validation':<20} {'Test':<20} {'Difference':<20}\n")
        f.write("-"*80 + "\n")
        
        for metric in ['Dice', 'IoU', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC', 'PR-AUC']:
            val_val = val_metrics[metric]
            test_val = test_metrics[metric]
            diff = test_val - val_val
            
            f.write(f"{metric:<20} {val_val:<20.4f} {test_val:<20.4f} {diff:<+20.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Validation Set:\n")
        f.write(f"  Dice:      {val_metrics['Dice']:.4f} ± {val_metrics['Dice_std']:.4f}\n")
        f.write(f"  IoU:       {val_metrics['IoU']:.4f} ± {val_metrics['IoU_std']:.4f}\n")
        f.write(f"  Precision: {val_metrics['Precision']:.4f}\n")
        f.write(f"  Recall:    {val_metrics['Recall']:.4f}\n")
        f.write(f"  F1-Score:  {val_metrics['F1-Score']:.4f}\n")
        f.write(f"  Accuracy:  {val_metrics['Accuracy']*100:.2f}%\n")
        f.write(f"  ROC-AUC:   {val_metrics['ROC-AUC']:.4f}\n")
        f.write(f"  PR-AUC:    {val_metrics['PR-AUC']:.4f}\n\n")
        
        f.write("Test Set:\n")
        f.write(f"  Dice:      {test_metrics['Dice']:.4f} ± {test_metrics['Dice_std']:.4f}\n")
        f.write(f"  IoU:       {test_metrics['IoU']:.4f} ± {test_metrics['IoU_std']:.4f}\n")
        f.write(f"  Precision: {test_metrics['Precision']:.4f}\n")
        f.write(f"  Recall:    {test_metrics['Recall']:.4f}\n")
        f.write(f"  F1-Score:  {test_metrics['F1-Score']:.4f}\n")
        f.write(f"  Accuracy:  {test_metrics['Accuracy']*100:.2f}%\n")
        f.write(f"  ROC-AUC:   {test_metrics['ROC-AUC']:.4f}\n")
        f.write(f"  PR-AUC:    {test_metrics['PR-AUC']:.4f}\n")


def main():
    print("="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    Path(CONFIG['results_dir']).mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    val_scans = load_dataset(CONFIG['val_path'])
    test_scans = load_dataset(CONFIG['test_path'])
    
    val_ds = VerseDataset(val_scans, num_slices=5, augment=False)
    test_ds = VerseDataset(test_scans, num_slices=5, augment=False)
    
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    # Load model
    print("\nLoading model...")
    model = UNet(in_channels=1, out_channels=2).to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
    
    # Collect predictions
    print("\n" + "="*70)
    print("VALIDATION SET EVALUATION")
    print("="*70)
    val_preds, val_targets, val_probs = collect_predictions(model, val_loader, CONFIG['device'])
    
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    test_preds, test_targets, test_probs = collect_predictions(model, test_loader, CONFIG['device'])
    
    # Calculate metrics
    print("\nCalculating metrics...")
    val_metrics = calculate_metrics(val_targets, val_preds, val_probs)
    test_metrics = calculate_metrics(test_targets, test_preds, test_probs)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    print("  1. Confusion matrices...")
    plot_confusion_matrix(val_targets, val_preds, 
                         f"{CONFIG['results_dir']}/confusion_matrix_val.png")
    plot_confusion_matrix(test_targets, test_preds,
                         f"{CONFIG['results_dir']}/confusion_matrix_test.png")
    
    print("  2. ROC curves...")
    plot_roc_curve(val_targets, val_probs,
                  f"{CONFIG['results_dir']}/roc_curve_val.png")
    plot_roc_curve(test_targets, test_probs,
                  f"{CONFIG['results_dir']}/roc_curve_test.png")
    
    print("  3. Precision-Recall curves...")
    plot_precision_recall_curve(val_targets, val_probs,
                               f"{CONFIG['results_dir']}/pr_curve_val.png")
    plot_precision_recall_curve(test_targets, test_probs,
                               f"{CONFIG['results_dir']}/pr_curve_test.png")
    
    print("  4. Metrics comparison...")
    plot_metrics_comparison(val_metrics, test_metrics,
                           f"{CONFIG['results_dir']}/metrics_comparison.png")
    
    # Save metrics table
    print("\nSaving metrics table...")
    save_metrics_table(val_metrics, test_metrics,
                      f"{CONFIG['results_dir']}/detailed_metrics.txt")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nTest Set Performance:")
    print(f"  Dice:      {test_metrics['Dice']:.4f} ± {test_metrics['Dice_std']:.4f}")
    print(f"  IoU:       {test_metrics['IoU']:.4f}")
    print(f"  Precision: {test_metrics['Precision']:.4f}")
    print(f"  Recall:    {test_metrics['Recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['F1-Score']:.4f}")
    print(f"  Accuracy:  {test_metrics['Accuracy']*100:.2f}%")
    print(f"  ROC-AUC:   {test_metrics['ROC-AUC']:.4f}")
    print(f"  PR-AUC:    {test_metrics['PR-AUC']:.4f}")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {CONFIG['results_dir']}/")
    print("\nGenerated files:")
    print("  - confusion_matrix_val.png")
    print("  - confusion_matrix_test.png")
    print("  - roc_curve_val.png")
    print("  - roc_curve_test.png")
    print("  - pr_curve_val.png")
    print("  - pr_curve_test.png")
    print("  - metrics_comparison.png")
    print("  - detailed_metrics.txt")
    print("="*70)


if __name__ == "__main__":
    main()
