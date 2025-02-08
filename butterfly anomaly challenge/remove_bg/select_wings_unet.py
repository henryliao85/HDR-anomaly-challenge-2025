#!/usr/bin/env python3
"""
This script loads a pre-trained U-Net model to remove backgrounds from images.
It reads image metadata from a CSV file, processes each image to remove its background,
and saves the processed images to an output folder.
Use segmentation scripts and pretrained SAM and YOLO networks to generate the training data
Usage:
    python remove_background.py --model_path path/to/model.pth \
                                --csv_path path/to/metadata.csv \
                                --output_folder path/to/output_images
"""

import os
import argparse

import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from train_unet256 import UNet256


def load_unet_model(model_path, device):
    """Load the U-Net model from the specified checkpoint."""
    model = UNet256()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def remove_background_color_based(image_np, mask_np):
    """
    Apply a binary mask to the image.
    Pixels where the mask is 255 are preserved; all others are replaced with white.
    """
    mask_bool = (mask_np == 255)
    masked_foreground = image_np.copy()
    masked_foreground[~mask_bool] = 0
    white_bg = np.full(image_np.shape, 255, dtype=np.uint8)
    white_bg[mask_bool] = 0
    final_image = masked_foreground + white_bg
    return Image.fromarray(final_image)


def remove_background_single(image_tensor: torch.Tensor, unet_model, device):
    """
    Remove the background of a single image tensor using the U-Net model.
    The image is resized to 256x256 and processed to produce a segmentation mask.
    """
    pil_img = transforms.ToPILImage()(image_tensor)
    resized_pil = pil_img.resize((256, 256), Image.BICUBIC)
    transform_seg = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform_seg(resized_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        mask_pred = unet_model(input_tensor)
    
    mask_prob = torch.sigmoid(mask_pred).squeeze().cpu().numpy()
    binary_mask = (mask_prob > 0.5).astype(np.uint8) * 255
    final_pil = remove_background_color_based(np.array(resized_pil), binary_mask)
    return transforms.ToTensor()(final_pil)


def process_and_save_images(csv_file, output_folder, unet_model, device):
    """
    For each row in the CSV file, load the corresponding image (from a folder
    specified in the CSV), remove its background, and save the resulting image.
    """
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(csv_file)
    
    for idx, row in df.iterrows():
        # Expect the CSV to have 'folder' and 'filename' columns.
        image_path = os.path.join(row['folder'], row['filename'])
        save_path = os.path.join(output_folder, row['filename'])
        
        if not os.path.exists(image_path):
            print(f"Skipping {image_path}, file not found.")
            continue
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open {image_path}: {e}")
            continue
        
        image_tensor = transforms.ToTensor()(image).to(device)
        wings_selected = remove_background_single(image_tensor, unet_model, device)
        final_image = transforms.ToPILImage()(wings_selected.cpu())
        final_image.save(save_path, format="png")
        print(f"Saved: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove image backgrounds using a pre-trained U-Net model."
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the U-Net model checkpoint (e.g. best_model.pth).")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to the CSV metadata file containing 'folder' and 'filename' columns.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Folder to save the processed images.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (default: cuda if available, else cpu).")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = args.device
    unet_model = load_unet_model(args.model_path, device)
    process_and_save_images(args.csv_path, args.output_folder, unet_model, device)

