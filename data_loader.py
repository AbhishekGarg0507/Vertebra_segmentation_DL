"""
Data loading utilities for VerSe19 dataset
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob


def load_dataset(folder_path):
    """Load CT and mask pairs from dataset folder"""
    ct_files = sorted(glob(f"{folder_path}/rawdata/sub-verse*/*_ct.nii.gz"))
    mask_files = sorted(glob(f"{folder_path}/derivatives/sub-verse*/*_seg-vert_msk.nii.gz"))
    
    ct_dict = {Path(f).name.split('_')[0].replace('sub-verse', ''): f for f in ct_files}
    mask_dict = {Path(f).name.split('_')[0].replace('sub-verse', ''): f for f in mask_files}
    
    pairs = []
    for subj_id in ct_dict.keys():
        if subj_id in mask_dict:
            pairs.append({'id': subj_id, 'ct': ct_dict[subj_id], 'mask': mask_dict[subj_id]})
    
    return pairs


def select_slices(mask_data, num_slices=10, min_pixels=500):
    """Select slices with most vertebrae content"""
    scores = [(i, np.sum(mask_data[:, :, i] > 0)) 
              for i in range(mask_data.shape[2]) 
              if np.sum(mask_data[:, :, i] > 0) > min_pixels]
    
    if not scores:
        return []
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    if len(scores) >= num_slices:
        top_slices = [s[0] for s in scores[:num_slices * 2]]
        top_slices.sort()
        indices = np.linspace(0, len(top_slices) - 1, num_slices, dtype=int)
        return [top_slices[i] for i in indices]
    
    return [s[0] for s in scores]


def visualize_samples(ct_path, mask_path, save_path, num_samples=3):
    """Create visualization of CT and mask slices"""
    ct_data = nib.load(ct_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()
    
    valid_slices = [i for i in range(mask_data.shape[2]) 
                    if np.sum(mask_data[:, :, i]) > 500]
    
    if not valid_slices:
        return
    
    sample_idx = np.linspace(0, len(valid_slices) - 1, 
                            min(num_samples, len(valid_slices)), dtype=int)
    slices = [valid_slices[i] for i in sample_idx]
    
    fig, axes = plt.subplots(len(slices), 3, figsize=(12, 4 * len(slices)))
    if len(slices) == 1:
        axes = [axes]
    
    for i, sl in enumerate(slices):
        ct_slice = ct_data[:, :, sl]
        mask_slice = mask_data[:, :, sl]
        ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
        
        axes[i, 0].imshow(ct_norm.T, cmap='gray', origin='lower')
        axes[i, 0].set_title(f'CT Slice {sl}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_slice.T, cmap='jet', origin='lower')
        axes[i, 1].set_title(f'Mask ({int(mask_slice.sum())} px)')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(ct_norm.T, cmap='gray', origin='lower')
        axes[i, 2].imshow(mask_slice.T, cmap='jet', alpha=0.4, origin='lower')
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
