"""
Main training script - Official VerSe19 split
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time

from data_loader import load_dataset, visualize_samples
from model import UNet
from dataset import VerseDataset, CombinedLoss
from training import train_epoch, validate, evaluate
from visualization import plot_training_curves, visualize_predictions, save_results


# Configuration
CONFIG = {
    'train_path': 'Dataset/dataset-verse19training',
    'val_path': 'Dataset/dataset-verse19validation',
    'test_path': 'Dataset/dataset-verse19test',
    'results_dir': 'results',
    'batch_size': 4,
    'num_epochs': 25,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'device': torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
}


def main():
    print("="*70)
    print("VERTEBRAE SEGMENTATION TRAINING")
    print("="*70)
    
    Path(CONFIG['results_dir']).mkdir(exist_ok=True)
    
    # Load data
    print("\nLoading datasets...")
    train_scans = load_dataset(CONFIG['train_path'])
    val_scans = load_dataset(CONFIG['val_path'])
    test_scans = load_dataset(CONFIG['test_path'])
    
    print(f"Training: {len(train_scans)} scans")
    print(f"Validation: {len(val_scans)} scans")
    print(f"Test: {len(test_scans)} scans")
    
    # Create sample visualizations
    visualize_samples(train_scans[0]['ct'], train_scans[0]['mask'], 
                     f"{CONFIG['results_dir']}/sample_train.png")
    visualize_samples(val_scans[0]['ct'], val_scans[0]['mask'], 
                     f"{CONFIG['results_dir']}/sample_val.png")
    visualize_samples(test_scans[0]['ct'], test_scans[0]['mask'], 
                     f"{CONFIG['results_dir']}/sample_test.png")
    
    # Create datasets
    print("\nCreating datasets...")
    train_ds = VerseDataset(train_scans, num_slices=10, augment=True)
    val_ds = VerseDataset(val_scans, num_slices=5, augment=False)
    test_ds = VerseDataset(test_scans, num_slices=5, augment=False)
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], 
                             shuffle=False, num_workers=0)
    
    # Initialize model
    print("\nInitializing model...")
    model = UNet(in_channels=1, out_channels=2).to(CONFIG['device'])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
                          weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)
    
    # Training
    print(f"\nTraining for {CONFIG['num_epochs']} epochs...")
    best_val_dice = 0
    train_losses, val_losses = [], []
    train_dice_list, val_dice_list = [], []
    
    start_time = time.time()
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        
        train_loss, train_dice = train_epoch(model, train_loader, criterion, 
                                            optimizer, CONFIG['device'])
        val_loss, val_dice, val_iou, val_acc = validate(model, val_loader, 
                                                         criterion, CONFIG['device'])
        
        scheduler.step(val_loss)
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), f"{CONFIG['results_dir']}/best_model.pth")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dice_list.append(train_dice)
        val_dice_list.append(val_dice)
        
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        print(f"Best Val Dice: {best_val_dice:.4f}")
    
    train_time = (time.time() - start_time) / 60
    print(f"\nTraining completed in {train_time:.1f} minutes")
    
    # Evaluation
    print("\nEvaluating model...")
    model.load_state_dict(torch.load(f"{CONFIG['results_dir']}/best_model.pth"))
    
    val_results = evaluate(model, val_loader, CONFIG['device'])
    test_results = evaluate(model, test_loader, CONFIG['device'])
    
    print("\nValidation Results:")
    print(f"Dice: {val_results['dice_mean']:.4f} ± {val_results['dice_std']:.4f}")
    print(f"IoU: {val_results['iou_mean']:.4f}")
    print(f"Accuracy: {val_results['acc_mean']*100:.2f}%")
    
    print("\nTest Results:")
    print(f"Dice: {test_results['dice_mean']:.4f} ± {test_results['dice_std']:.4f}")
    print(f"IoU: {test_results['iou_mean']:.4f}")
    print(f"Accuracy: {test_results['acc_mean']*100:.2f}%")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_training_curves(train_losses, val_losses, train_dice_list, val_dice_list,
                        f"{CONFIG['results_dir']}/training_curves.png")
    
    visualize_predictions(model, val_loader, CONFIG['device'], 6,
                         f"{CONFIG['results_dir']}/predictions_val.png")
    
    visualize_predictions(model, test_loader, CONFIG['device'], 6,
                         f"{CONFIG['results_dir']}/predictions_test.png")
    
    # Save results
    model_info = {
        'params': n_params,
        'time': train_time,
        'epochs': CONFIG['num_epochs'],
        'train_scans': len(train_scans),
        'val_scans': len(val_scans),
        'test_scans': len(test_scans)
    }
    
    save_results({'validation': val_results, 'test': test_results}, 
                model_info, f"{CONFIG['results_dir']}/results.txt")
    
    print(f"\nResults saved to {CONFIG['results_dir']}/")
    print("\nGenerated files:")
    print("  - sample_train.png, sample_val.png, sample_test.png")
    print("  - training_curves.png")
    print("  - predictions_val.png, predictions_test.png")
    print("  - results.txt")
    print("  - best_model.pth")
    print("="*70)


if __name__ == "__main__":
    main()
