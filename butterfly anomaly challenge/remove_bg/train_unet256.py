#!/usr/bin/env python3
"""
Train U-Net for binary wing segmentation.

Usage:
    python train_unet.py --train_image_dir path/to/train_images \
                         --train_mask_dir path/to/train_masks \
                         --val_image_dir path/to/val_images \
                         --val_mask_dir path/to/val_masks \
                         [--epochs 50] [--batch_size 16] [--lr 1e-4] \
                         [--output_dir path/to/output] [--no_cuda]
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# =========================
# 1. Dataset Handling
# =========================
class WingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        
        # List image files (png, jpg, jpeg)
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        # Verify mask existence and pairing based on a naming convention.
        self.mask_files = []
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            # For example, if mask files follow a convention like: {basename}_mask.png
            mask_file = f"{base_name}_mask.png"
            mask_path = os.path.join(mask_dir, mask_file)
            if os.path.exists(mask_path):
                self.mask_files.append(mask_file)
            else:
                raise FileNotFoundError(f"Mask for {img_file} not found at {mask_path}")

        assert len(self.image_files) == len(self.mask_files), (
            f"Mismatch: {len(self.image_files)} images vs {len(self.mask_files)} masks"
        )

        # Define transform for images.
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image.
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        
        # Load mask and process.
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = Image.open(mask_path).convert("L")
        # Convert mask to tensor in range [0,255]
        mask = (transforms.ToTensor()(mask) * 255).byte()
        
        # Convert multi-class mask to a binary wing mask:
        # For example, treating pixel values {1,2,3,4} as wings (set to 1), rest as background.
        mask = np.isin(mask.numpy(), [1, 2, 3, 4]).astype(np.float32)
        
        # Apply image transform.
        image = self.image_transform(image)
        # Convert mask to tensor.
        mask = torch.tensor(mask, dtype=torch.float32)
        
        return image, mask


# =========================
# 2. U-Net Model
# =========================
class DoubleConv(nn.Module):
    """Two consecutive Conv2d layers with BatchNorm and ReLU activation."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class UNet256(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        
        # Bridge
        self.bridge = DoubleConv(512, 1024)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        
        # Output layer (raw logits, use with BCEWithLogitsLoss)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)          # 256x256 -> 256x256
        p1 = nn.MaxPool2d(2)(c1)      # 256x256 -> 128x128
        
        c2 = self.down2(p1)         # 128x128 -> 128x128
        p2 = nn.MaxPool2d(2)(c2)      # 128x128 -> 64x64
        
        c3 = self.down3(p2)         # 64x64 -> 64x64
        p3 = nn.MaxPool2d(2)(c3)      # 64x64 -> 32x32
        
        c4 = self.down4(p3)         # 32x32 -> 32x32
        p4 = nn.MaxPool2d(2)(c4)      # 32x32 -> 16x16
        
        # Bridge
        b = self.bridge(p4)         # 16x16 -> 16x16
        
        # Decoder
        u1 = self.up1(b)            # 16x16 -> 32x32
        u1 = torch.cat([u1, c4], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)           # 32x32 -> 64x64
        u2 = torch.cat([u2, c3], dim=1)
        u2 = self.conv_up2(u2)
        
        u3 = self.up3(u2)           # 64x64 -> 128x128
        u3 = torch.cat([u3, c2], dim=1)
        u3 = self.conv_up3(u3)
        
        u4 = self.up4(u3)           # 128x128 -> 256x256
        u4 = torch.cat([u4, c1], dim=1)
        u4 = self.conv_up4(u4)
        
        return self.out(u4)         # Output raw logits


# =========================
# 3. Training Loop
# =========================
def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    # Create datasets
    train_dataset = WingDataset(
        image_dir=args.train_image_dir,
        mask_dir=args.train_mask_dir,
        image_size=256
    )
    val_dataset = WingDataset(
        image_dir=args.val_image_dir,
        mask_dir=args.val_mask_dir,
        image_size=256
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model setup
    model = UNet256().to(device)
    
    # Loss, optimizer, and learning rate scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        # Training loop
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))  # Add channel dimension to mask
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks.unsqueeze(1))
                val_loss += loss.item() * images.size(0)
        
        # Compute average losses
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.8f} | Val Loss = {val_loss:.8f}")
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print("  [*] Saved new best model")

# =========================
# 4. Argparse Setup
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for binary wing segmentation")
    
    # Data paths
    parser.add_argument("--train_image_dir", type=str, required=True,
                        help="Directory containing training images.")
    parser.add_argument("--train_mask_dir", type=str, required=True,
                        help="Directory containing training masks.")
    parser.add_argument("--val_image_dir", type=str, required=True,
                        help="Directory containing validation images.")
    parser.add_argument("--val_mask_dir", type=str, required=True,
                        help="Directory containing validation masks.")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    
    # Output and device settings
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save the best model.")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(args)

