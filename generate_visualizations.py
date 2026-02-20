"""
Generate visualizations from trained model
Run this AFTER training is complete
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from data_loader import load_dataset
from model import UNet
from dataset import VerseDataset
from training import evaluate
from visualization import plot_training_curves, visualize_predictions, save_results


# Configuration - UPDATE THESE TO MATCH YOUR TRAINING
CONFIG = {
    'train_path': 'Dataset/dataset-verse19training',
    'val_path': 'Dataset/dataset-verse19validation',
    'test_path': 'Dataset/dataset-verse19test',
    'results_dir': 'results',
    'model_path': 'results/best_model.pth',  # Your trained model
    'batch_size': 4,
    'device': torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
}


def main():
    print("="*70)
    print("GENERATING VISUALIZATIONS FROM TRAINED MODEL")
    print("="*70)
    
    # Check if model exists
    if not Path(CONFIG['model_path']).exists():
        print(f"\nError: Model not found at {CONFIG['model_path']}")
        print("Please train the model first or update the model_path")
        return
    
    # Load datasets
    print("\nLoading datasets...")
    val_scans = load_dataset(CONFIG['val_path'])
    test_scans = load_dataset(CONFIG['test_path'])
    
    print(f"Validation: {len(val_scans)} scans")
    print(f"Test: {len(test_scans)} scans")
    
    # Create datasets
    print("\nCreating datasets...")
    val_ds = VerseDataset(val_scans, num_slices=5, augment=False)
    test_ds = VerseDataset(test_scans, num_slices=5, augment=False)
    
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], 
                             shuffle=False, num_workers=0)
    
    # Load model
    print("\nLoading trained model...")
    model = UNet(in_channels=1, out_channels=2).to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
    print("Model loaded successfully")
    
    # Evaluate
    print("\nEvaluating on validation set...")
    val_results = evaluate(model, val_loader, CONFIG['device'])
    
    print("\nEvaluating on test set...")
    test_results = evaluate(model, test_loader, CONFIG['device'])
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print("\nValidation Set:")
    print(f"  Dice: {val_results['dice_mean']:.4f} ± {val_results['dice_std']:.4f}")
    print(f"  IoU:  {val_results['iou_mean']:.4f}")
    print(f"  Acc:  {val_results['acc_mean']*100:.2f}%")
    
    print("\nTest Set:")
    print(f"  Dice: {test_results['dice_mean']:.4f} ± {test_results['dice_std']:.4f}")
    print(f"  IoU:  {test_results['iou_mean']:.4f}")
    print(f"  Acc:  {test_results['acc_mean']*100:.2f}%")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\nCreating prediction visualizations...")
    visualize_predictions(model, val_loader, CONFIG['device'], 6,
                         f"{CONFIG['results_dir']}/predictions_val.png")
    print("  ✓ Saved predictions_val.png")
    
    visualize_predictions(model, test_loader, CONFIG['device'], 6,
                         f"{CONFIG['results_dir']}/predictions_test.png")
    print("  ✓ Saved predictions_test.png")
    
    # Save results to text file
    print("\nSaving results...")
    model_info = {
        'params': sum(p.numel() for p in model.parameters()),
        'time': 0,  # Not tracked in this script
        'epochs': 25,  # Update if different
        'train_scans': len(load_dataset(CONFIG['train_path'])),
        'val_scans': len(val_scans),
        'test_scans': len(test_scans)
    }
    
    save_results({'validation': val_results, 'test': test_results}, 
                model_info, f"{CONFIG['results_dir']}/results.txt")
    print("  ✓ Saved results.txt")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nGenerated files in '{CONFIG['results_dir']}/':")
    print("  - predictions_val.png")
    print("  - predictions_test.png")
    print("  - results.txt")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
