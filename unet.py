import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)


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


        
    

