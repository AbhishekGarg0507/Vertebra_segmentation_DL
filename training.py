"""
Training and validation functions
"""

import torch
from tqdm import tqdm


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_dice = 0
    
    for imgs, masks in tqdm(loader, desc='Training', leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            dice = (preds == masks).float().mean()
            total_dice += dice.item()
    
    return total_loss / len(loader), total_dice / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_acc = 0
    
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Validation', leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            preds = torch.argmax(outputs, dim=1)
            
            total_loss += loss.item()
            
            pred_f = preds.view(-1).float()
            mask_f = masks.view(-1).float()
            inter = (pred_f * mask_f).sum()
            union_dice = pred_f.sum() + mask_f.sum()
            union_iou = union_dice - inter
            
            total_dice += (2 * inter + 1) / (union_dice + 1)
            total_iou += (inter + 1) / (union_iou + 1)
            total_acc += (preds == masks).float().mean().item()
    
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n, total_acc / n


def evaluate(model, loader, device):
    model.eval()
    dice_scores = []
    iou_scores = []
    acc_scores = []
    
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Evaluating', leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            
            for pred, mask in zip(preds, masks):
                pred_f = pred.view(-1).float()
                mask_f = mask.view(-1).float()
                inter = (pred_f * mask_f).sum()
                union = pred_f.sum() + mask_f.sum()
                
                dice = (2 * inter + 1) / (union + 1)
                iou = (inter + 1) / (union - inter + 1)
                acc = (pred == mask).float().mean()
                
                dice_scores.append(dice.item())
                iou_scores.append(iou.item())
                acc_scores.append(acc.item())
    
    import numpy as np
    return {
        'dice_mean': np.mean(dice_scores),
        'dice_std': np.std(dice_scores),
        'iou_mean': np.mean(iou_scores),
        'acc_mean': np.mean(acc_scores)
    }
