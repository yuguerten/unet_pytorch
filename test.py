import os
import cv2
import torch
import numpy as np
import argparse
from train import Unet  # re-use the exact architecture in train.py

def preprocess(img_path, size):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    # train.py uses (width, height) ordering
    w, h = size
    img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    # convert BGR->RGB and scale to [0,1]
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float()  # 1x3xH xW
    return tensor, img_resized

def postprocess(probs, thr=0.5):
    mask = (probs >= thr).astype(np.uint8) * 255
    return mask

def overlay_bgr(orig_bgr, mask_bin, color=(0,0,255), alpha=0.4):
    color_layer = np.zeros_like(orig_bgr)
    color_layer[mask_bin == 255] = color
    return cv2.addWeighted(orig_bgr, 1.0 - alpha, color_layer, alpha, 0)

def load_model(path, device):
    model = Unet(in_ch=3, out_ch=1).to(device)
    state = torch.load(path, map_location=device)
    # state should be a state_dict saved with torch.save(model.state_dict())
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        # strip module. if saved from DataParallel
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
        state = new_state
    model.load_state_dict(state)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--model", "-m", default="unet_model.pth", help="Model .pth path (state_dict)")
    parser.add_argument("--out_mask", default="pred_mask.png", help="Output mask path")
    parser.add_argument("--out_overlay", default="pred_overlay.png", help="Output overlay path")
    parser.add_argument("--size", "-s", nargs=2, type=int, default=[512, 512],
                        help="Resize size (width height). train.py used 512x512 by default")
    parser.add_argument("--device", "-d", default=None, help="cuda or cpu (auto if omitted)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Binarization threshold")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)
    if not os.path.exists(args.model):
        raise FileNotFoundError(args.model)

    input_tensor, resized_bgr = preprocess(args.image, tuple(args.size))
    input_tensor = input_tensor.to(device, non_blocking=True)

    model = load_model(args.model, device)

    with torch.no_grad():
        logits = model(input_tensor)  # 1x1xHxW
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # HxW

    mask_bin = postprocess(probs, thr=args.threshold)

    cv2.imwrite(args.out_mask, mask_bin)
    overlay = overlay_bgr(resized_bgr, mask_bin, color=(0,0,255), alpha=0.4)
    cv2.imwrite(args.out_overlay, overlay)

    print(f"Saved mask -> {args.out_mask}")
    print(f"Saved overlay -> {args.out_overlay}")

if __name__ == "__main__":
    main()