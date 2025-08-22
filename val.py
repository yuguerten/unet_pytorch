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

def visualize_predictions(model, val_loader, device, num_samples=5):
    """Visualize some predictions"""
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
            
            fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
            if batch_size == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(batch_size):
                # Original image (convert CHW to HWC)
                img = np.transpose(images[i], (1, 2, 0))
                axes[i, 0].imshow(img)
                axes[i, 0].set_title('Original Image')
                axes[i, 0].axis('off')
                
                # True mask
                axes[i, 1].imshow(true_masks[i, 0], cmap='gray')
                axes[i, 1].set_title('True Mask')
                axes[i, 1].axis('off')
                
                # Predicted mask
                axes[i, 2].imshow(pred_masks[i, 0], cmap='gray')
                axes[i, 2].set_title('Predicted Mask')
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
        model.load_state_dict(torch.load(model_path, map_location=device))
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
    model_path = "unet_model.pth"
    val_folder_path = "/home/ouaaziz/workplace/unet_pytorch/dataset/val"
    
    results = validate_model(model_path, val_folder_path)
    print(f"\nFinal Results:")
    print(f"Validation Loss: {results['val_loss']:.4f}")
    print(f"IoU Score: {results['iou']:.4f}")
    print(f"Dice Score: {results['dice']:.4f}")
