import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
import logging
import glob
import os
import cv2
logger = logging.getLogger(__name__)

class BrainTumorDataset(Dataset):
    def __init__(self, folder_path):
        super().__init__()
        self.img_files = glob.glob(os.path.join(folder_path, 'images', '*.jpg'))
        self.mask_files = []
        
        # Filter out images that don't have corresponding masks
        valid_pairs = []
        for img_path in self.img_files:
            mask_path = os.path.join(folder_path, 'masks', os.path.basename(img_path))
            if os.path.exists(mask_path):
                valid_pairs.append((img_path, mask_path))
            else:
                print(f"Warning: Mask not found for {img_path}")
        
        self.img_files = [pair[0] for pair in valid_pairs]
        self.mask_files = [pair[1] for pair in valid_pairs]

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        
        # Read images and check if they loaded successfully
        data = cv2.imread(img_path)
        label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Masks are typically grayscale
        
        if data is None:
            raise ValueError(f"Could not load image: {img_path}")
        if label is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Convert BGR to RGB for the image
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        data = data.astype(np.float32) / 255.0
        label = label.astype(np.float32) / 255.0
        
        # Convert to tensors and fix dimensions
        # Image: (H, W, C) -> (C, H, W)
        data = torch.from_numpy(data).permute(2, 0, 1)
        # Mask: (H, W) -> (H, W) for single class or convert to proper format
        label = torch.from_numpy(label)
        
        return data, label
    
    def __len__(self):
        return len(self.img_files)

# ----------------- Encoder block --------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
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

        self.final_conv = nn.Conv2d(prev_ch, out_ch, kernel_size=1)

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
    folder_path = "/home/yuguerten/workspace/unet_pytorch/dataset"
    brain_tunor_dataset = BrainTumorDataset(folder_path)
    train_loader = DataLoader(brain_tunor_dataset, batch_size=16, shuffle=True)
    model = Unet(in_ch=3, out_ch=1)
    optimizer = Adam(model.parameters())
    loss_fct = nn.BCEWithLogitsLoss()
    
    model.train()

    for epoch in range(2):
        total_loss = 0
        current_loss = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            output = output.squeeze(1)

            loss = loss_fct(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            current_loss += 1
        
        print(f"Epoch: {epoch}, Total Loss: {total_loss/current_loss}")
            



        
    

