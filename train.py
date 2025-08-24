import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms
import numpy as np
import logging
import glob
import os
import cv2
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class BrainTumorDataset(Dataset):
    def __init__(self, folder_path, image_size=(512, 512)):
        super().__init__()
        self.folder_path = folder_path
        self.image_size = image_size  # (width, height)
        self.img_files = glob.glob(os.path.join(folder_path, 'images', '*.jpg'))
        valid_pairs = []
        for img_path in self.img_files:
            mask_path = os.path.join(folder_path, 'masks', os.path.basename(img_path))
            if os.path.exists(mask_path):
                valid_pairs.append((img_path, mask_path))
            else:
                print(f"Warning: Mask not found for {img_path}")
        self.img_files = [p[0] for p in valid_pairs]
        self.mask_files = [p[1] for p in valid_pairs]

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        data = cv2.imread(img_path)
        label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if data is None:
            raise ValueError(f"Could not load image: {img_path}")
        if label is None:
            raise ValueError(f"Could not load mask: {mask_path}")

        # Resize to fixed size (width, height)
        w, h = self.image_size
        data = cv2.resize(data, (w, h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        # BGR -> RGB
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        # Normalize to [0,1]
        data = data.astype(np.float32) / 255.0
        label = label.astype(np.float32) / 255.0

        # HWC -> CHW
        data = torch.from_numpy(data).permute(2, 0, 1).float()
        # add channel dim for mask: (H, W) -> (1, H, W)
        label = torch.from_numpy(label).unsqueeze(0).float()

        return data, label

    def __len__(self):
        return len(self.img_files)

# ----------------- Encoder block --------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

# ----------------- Decoder block --------------------------
class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        # logger.info(x.shape)
        x = self.up(x)
        # logger.info(x.shape)
        # logger.info(skip.shape)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv(x)
        return x
    
# ----------------- Unet --------------------------
class Unet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.feature_channels = [32, 64, 128, 256, 512]
        self.encoders = nn.ModuleList()
        prev_ch = in_ch

        for feat_ch in self.feature_channels:
            self.encoders.append(DoubleConv(prev_ch, feat_ch))
            prev_ch = feat_ch
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # bottleneck 
        self.bottleneck = DoubleConv(prev_ch, prev_ch * 2)

        self.decoders = nn.ModuleList()
        prev_ch = self.feature_channels[-1] * 2
        for feat_ch in reversed(self.feature_channels):
            self.decoders.append(Up(prev_ch, feat_ch, feat_ch))
            prev_ch = feat_ch

        self.final_conv = nn.Conv2d(prev_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
        
        x = self.final_conv(x)
        return x

if __name__ == "__main__":
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    train_folder_path = "dataset/train"
    val_folder_path = "dataset/val"

    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    IN_CH = 3
    OUT_CH = 1
    EPOCHS = 100

    torch.backends.cudnn.benchmark = True  # speed autotune for fixed size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    brain_tumor_train_dataset = BrainTumorDataset(train_folder_path)
    brain_tumor_val_dataset = BrainTumorDataset(val_folder_path)
    logger.info(f"Train dataset size: {len(brain_tumor_train_dataset)}")
    logger.info(f"Validation dataset size: {len(brain_tumor_val_dataset)}")
    
    train_loader = DataLoader(
        brain_tumor_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        brain_tumor_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    model = Unet(in_ch=IN_CH, out_ch=OUT_CH).to(device).to(memory_format=torch.channels_last)
    try:
        model = torch.compile(model)  # PyTorch 2.x
    except Exception:
        logger.info("torch.compile not used (unsupported environment).")

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    # If supported: from torch.optim import AdamW; optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, fused=True)
    loss_fct = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info("Starting training...")

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.time()
        total_train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{EPOCHS}', unit='batch', dynamic_ncols=True)
        
        for batch_idx, (data, label) in enumerate(pbar):
            data = data.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            label = label.to(device, non_blocking=True)

            for param in model.parameters():
                param.grad = None

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model(data)
                loss = loss_fct(output, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_train_loss/(batch_idx+1):.4f}',
                'GPU Mem': f'{torch.cuda.memory_allocated()/1e9:.2f}GB' if torch.cuda.is_available() else 'N/A'
            })
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{EPOCHS}', unit='batch', dynamic_ncols=True)
            for batch_idx, (data, label) in enumerate(val_pbar):
                data = data.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                label = label.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    output = model(data)
                    loss = loss_fct(output, label)
                total_val_loss += loss.item()
                val_pbar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'Avg Val Loss': f'{total_val_loss/(batch_idx+1):.4f}'
                })
        
        avg_val_loss = total_val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1}/{EPOCHS} took {epoch_time:.2f}s - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")
    torch.save(model.state_dict(), 'unet_model.pth')
    logger.info("Model saved as 'unet_model.pth'")







