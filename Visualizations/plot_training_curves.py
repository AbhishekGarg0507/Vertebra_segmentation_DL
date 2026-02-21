"""
Generate training curves from saved training history
Use this if you saved your losses and dice scores during training
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np


def plot_curves_from_data(train_losses, val_losses, train_dice, val_dice, save_path='results/training_curves.png'):
    """
    Plot training curves from saved data
    
    Args:
        train_losses: list of training losses per epoch
        val_losses: list of validation losses per epoch
        train_dice: list of training dice scores per epoch
        val_dice: list of validation dice scores per epoch
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train', marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='Validation', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Dice plot
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
    print(f"Saved training curves to {save_path}")
    plt.close()


# Your actual training data from the output
train_losses = [0.4680, 0.3376, 0.2792, 0.2545, 0.2372, 0.2196, 0.2093, 0.1875, 0.1755, 0.1544,
                0.1394, 0.1253, 0.1303, 0.1176, 0.1084, 0.1038, 0.1065, 0.0926, 0.0910, 0.0921,
                0.0918, 0.0907, 0.0790, 0.0698, 0.0687]

val_losses = [0.3881, 0.3295, 0.2910, 0.3363, 0.3778, 0.3909, 0.2358, 0.3418, 0.5031, 0.1545,
              0.2421, 0.1777, 0.1535, 0.1266, 0.1726, 0.1221, 0.1146, 0.1345, 0.1337, 0.1237,
              0.2438, 0.1478, 0.0996, 0.0920, 0.0991]

train_dice = [0.9198, 0.9442, 0.9541, 0.9585, 0.9614, 0.9646, 0.9666, 0.9703, 0.9728, 0.9762,
              0.9786, 0.9808, 0.9805, 0.9822, 0.9835, 0.9842, 0.9839, 0.9859, 0.9861, 0.9858,
              0.9861, 0.9863, 0.9881, 0.9892, 0.9894]

val_dice = [0.5826, 0.6394, 0.6854, 0.6170, 0.5533, 0.5317, 0.7429, 0.6183, 0.4213, 0.8393,
            0.7318, 0.8092, 0.8382, 0.8659, 0.8160, 0.8709, 0.8791, 0.8620, 0.8587, 0.8680,
            0.7345, 0.8451, 0.8945, 0.9020, 0.8951]


if __name__ == "__main__":
    print("="*70)
    print("GENERATING TRAINING CURVES")
    print("="*70)
    
    print("\nUsing your actual training data from 25 epochs")
    print("Training time: 265.3 minutes")
    print("Best validation Dice: 0.9020\n")
    
    # Check if all lists have same length
    if not (len(train_losses) == len(val_losses) == len(train_dice) == len(val_dice)):
        print("Error: All lists must have the same length!")
        print(f"train_losses: {len(train_losses)}")
        print(f"val_losses: {len(val_losses)}")
        print(f"train_dice: {len(train_dice)}")
        print(f"val_dice: {len(val_dice)}")
    else:
        plot_curves_from_data(train_losses, val_losses, train_dice, val_dice)
        print("\n✓ Done!")


# OPTION 2: If you saved the data to a file during training
# Uncomment and use this instead:

# with open('training_history.pkl', 'rb') as f:
#     history = pickle.load(f)
# 
# plot_curves_from_data(
#     history['train_losses'],
#     history['val_losses'],
#     history['train_dice'],
#     history['val_dice']
# )
