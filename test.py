from unet import Unet
import torch
import logging
logging.basicConfig(level=logging.INFO, force=True)

if __name__ == "__main__":
    model = Unet(1, out_ch=3)
    input = torch.rand(1, 1, 256, 256)
    output = model(input)
    print(input.shape)
    print(output.shape)      