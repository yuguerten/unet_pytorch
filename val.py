import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from train import BrainTumorDataset, Unet
import os
import cv2

def apply_color_mask(image, mask, color=(255, 0, 0), alpha=0.4, threshold=0.5):
    """Return an RGB image with a colored transparent overlay where mask==1.

    Args:
        image (np.ndarray): Original image in range [0,1] shape (H,W,3)
        mask (np.ndarray): Predicted (or true) mask probabilities shape (H,W) or (1,H,W)
        color (tuple): BGR color (as OpenCV uses) or RGB? We'll accept RGB then convert.
        alpha (float): Transparency of overlay color.
        threshold (float): Threshold to binarize predicted mask.
    """
    if mask.ndim == 3:
        mask = mask[0]
    # Binarize
    bin_mask = (mask >= threshold).astype(np.uint8)
    # Ensure image float
    img = image.copy()
    img = np.clip(img, 0, 1)
    # Create color layer (convert RGB->BGR for cv2 blending if using cv2)
    overlay = (np.array(color).reshape(1,1,3) / 255.0).astype(np.float32)
    overlay_img = img.copy()
    overlay_img[bin_mask == 1] = (1 - alpha) * overlay_img[bin_mask == 1] + alpha * overlay
    return overlay_img, bin_mask

def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union (IoU) metric"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = torch.sum(pred_binary * target_binary, dim=(2, 3))
    union = torch.sum(pred_binary + target_binary - pred_binary * target_binary, dim=(2, 3))
    
    # Avoid division by zero
    iou = intersection / (union + 1e-8)
    return iou.mean()

def calculate_dice(pred, target, threshold=0.5):
    """Calculate Dice coefficient"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = torch.sum(pred_binary * target_binary, dim=(2, 3))
    dice = (2. * intersection) / (torch.sum(pred_binary, dim=(2, 3)) + torch.sum(target_binary, dim=(2, 3)) + 1e-8)
    
    return dice.mean()

def visualize_predictions(model, val_loader, device, num_samples=5, threshold=0.5):
    """Visualize some predictions with colored overlays.

    Saves figures like validation_samples_<idx>.png containing:
        Column 1: Original image
        Column 2: True mask overlay (green)
        Column 3: Predicted mask overlay (red)
        Column 4: Side-by-side true (green) + predicted (red edges) optional (only if binary mask present)
    """
    model.eval()
    samples_shown = 0
    
    with torch.no_grad():
        for data, label in val_loader:
            if samples_shown >= num_samples:
                break
                
            data = data.to(device)
            label = label.to(device)
            
            output = model(data)
            pred_masks = torch.sigmoid(output)
            
            # Convert to CPU and numpy for visualization
            images = data.cpu().numpy()
            true_masks = label.cpu().numpy()
            pred_masks = pred_masks.cpu().numpy()
            
            batch_size = min(data.size(0), num_samples - samples_shown)
            
            fig, axes = plt.subplots(batch_size, 3, figsize=(15, 4 * batch_size))
            if batch_size == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(batch_size):
                # Original image (convert CHW to HWC)
                img = np.transpose(images[i], (1, 2, 0))
                axes[i, 0].imshow(img)
                axes[i, 0].set_title('Original')
                axes[i, 0].axis('off')

                # True mask overlay (green)
                true_overlay, _ = apply_color_mask(img, true_masks[i], color=(0, 255, 0), alpha=0.45, threshold=threshold)
                axes[i, 1].imshow(true_overlay)
                axes[i, 1].set_title('True Overlay')
                axes[i, 1].axis('off')

                # Predicted mask overlay (red)
                pred_overlay, pred_bin = apply_color_mask(img, pred_masks[i], color=(255, 0, 0), alpha=0.45, threshold=threshold)
                axes[i, 2].imshow(pred_overlay)
                axes[i, 2].set_title('Pred Overlay')
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'validation_samples_{samples_shown}.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            samples_shown += batch_size

def validate_model(model_path, val_folder_path):
    """Main validation function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('validation.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load dataset
    val_dataset = BrainTumorDataset(val_folder_path)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Load model
    model = Unet(in_ch=3, out_ch=1).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.error(f"Model file {model_path} not found!")
        return
    
    model.eval()
    
    # Validation metrics
    total_val_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_batches = len(val_loader)
    
    loss_fct = nn.BCEWithLogitsLoss()
    
    logger.info("Starting validation...")
    start_time = time.time()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating', unit='batch', dynamic_ncols=True)
        
        for batch_idx, (data, label) in enumerate(pbar):
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            
            output = model(data)
            
            # Calculate loss
            loss = loss_fct(output, label)
            total_val_loss += loss.item()
            
            # Calculate metrics
            iou = calculate_iou(output, label)
            dice = calculate_dice(output, label)
            
            total_iou += iou.item()
            total_dice += dice.item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{iou.item():.4f}',
                'Dice': f'{dice.item():.4f}'
            })
    
    # Calculate average metrics
    avg_val_loss = total_val_loss / num_batches
    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches
    
    validation_time = time.time() - start_time
    
    logger.info(f"Validation completed in {validation_time:.2f}s")
    logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")
    logger.info(f"Average IoU: {avg_iou:.4f}")
    logger.info(f"Average Dice Score: {avg_dice:.4f}")
    
    # Visualize some predictions
    logger.info("Generating visualization samples...")
    visualize_predictions(model, val_loader, device, num_samples=5)
    
    return {
        'val_loss': avg_val_loss,
        'iou': avg_iou,
        'dice': avg_dice
    }

if __name__ == "__main__":
    model_path = "/home/ouaaziz/workplace/unet_pytorch/models/unet_model_4.pth"
    val_folder_path = "dataset/val"
    
    results = validate_model(model_path, val_folder_path)
    print(f"\nFinal Results:")
    print(f"Validation Loss: {results['val_loss']:.4f}")
    print(f"IoU Score: {results['iou']:.4f}")
    print(f"Dice Score: {results['dice']:.4f}")
