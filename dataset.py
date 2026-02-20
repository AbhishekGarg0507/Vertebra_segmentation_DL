"""
Dataset and loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from tqdm import tqdm


class VerseDataset(Dataset):
    def __init__(self, scan_list, num_slices=10, img_size=256, augment=False, min_pixels=500):
        self.img_size = img_size
        self.augment = augment
        self.samples = []
        
        for scan in tqdm(scan_list, desc='Loading data'):
            try:
                mask = nib.load(scan['mask']).get_fdata()
                slices = self._select_slices(mask, num_slices, min_pixels)
                
                if len(slices) >= 3:
                    for sl in slices:
                        self.samples.append((scan['ct'], scan['mask'], sl))
            except:
                pass
    
    def _select_slices(self, mask, n, min_px):
        scores = [(i, np.sum(mask[:, :, i] > 0)) 
                  for i in range(mask.shape[2]) 
                  if np.sum(mask[:, :, i] > 0) > min_px]
        
        if not scores:
            return []
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if len(scores) >= n:
            top = [s[0] for s in scores[:n * 2]]
            top.sort()
            idx = np.linspace(0, len(top) - 1, n, dtype=int)
            return [top[i] for i in idx]
        
        return [s[0] for s in scores]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ct_path, mask_path, slice_idx = self.samples[idx]
        
        ct = nib.load(ct_path).get_fdata()[:, :, slice_idx]
        mask = nib.load(mask_path).get_fdata()[:, :, slice_idx]
        
        ct_min, ct_max = ct.min(), ct.max()
        if ct_max > ct_min:
            ct = (ct - ct_min) / (ct_max - ct_min)
        
        mask = (mask > 0).astype(np.float32)
        
        ct_t = torch.from_numpy(ct).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        
        ct_t = F.interpolate(ct_t.unsqueeze(0), size=(self.img_size, self.img_size),
                            mode='bilinear', align_corners=False).squeeze(0)
        mask_t = F.interpolate(mask_t.unsqueeze(0), size=(self.img_size, self.img_size),
                              mode='nearest').squeeze(0)
        
        if self.augment:
            if torch.rand(1) > 0.5:
                ct_t = torch.flip(ct_t, dims=[2])
                mask_t = torch.flip(mask_t, dims=[2])
            if torch.rand(1) > 0.5:
                ct_t = ct_t * (0.8 + 0.4 * torch.rand(1))
                ct_t = torch.clamp(ct_t, 0, 1)
        
        return ct_t, mask_t.long().squeeze(0)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)[:, 1]
        target = target.float()
        inter = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1 - (2 * inter + 1) / (union + 1)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, pred, target):
        return 0.7 * self.dice(pred, target) + 0.3 * self.ce(pred, target)


def dice_score(pred, target):
    pred = pred.view(-1).float()
    target = target.view(-1).float()
    inter = (pred * target).sum()
    return (2 * inter + 1) / (pred.sum() + target.sum() + 1)


def iou_score(pred, target):
    pred = pred.view(-1).float()
    target = target.view(-1).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + 1) / (union + 1)
