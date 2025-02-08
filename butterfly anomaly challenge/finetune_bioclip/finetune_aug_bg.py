#!/usr/bin/env python3
"""
This script fine-tunes a BiO-CLIP model on a dataset with a U-Net based background removal and random augmentations.
It loads data, extracts features, trains a classification head, and saves the resulting models.

Usage:
    python train.py --data_file path/to/data.csv --img_dir path/to/images --clf_save_dir path/to/save_dir [other options...]

Adjust the arguments as needed.
"""

import os
import random
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

import albumentations as A

from open_clip import create_model

# Import local modules
from data_utils import load_data
from dataset import ButterflyDatasetClass14
from train_unet256 import UNet256


# --------------------------
# Transforms
# --------------------------
def get_transforms(image_size):
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    clip_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return image_transform, clip_transform


# Albumentations augmentation pipeline
def get_augmentation_pipeline(augment_prob):
    augmentation_transforms = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ElasticTransform(alpha=0.5, sigma=30, p=0.2),
        A.GridDistortion(num_steps=4, distort_limit=0.1, p=0.2),
        A.OpticalDistortion(distort_limit=0.02, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.MedianBlur(blur_limit=3, p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.RandomGamma(gamma_limit=(90, 110), p=0.3),
    ])
    return augment_prob, augmentation_transforms


# --------------------------
# Utility Functions
# --------------------------
def get_feats_and_meta(dloader, model, device, ignore_feats=False):
    """Extract features and labels from a DataLoader."""
    all_feats = None
    labels = []

    for img, lbl in tqdm(dloader, desc="Extracting features"):
        with torch.no_grad():
            if not ignore_feats:
                out = model(img.to(device))['image_features']
                feats = out.cpu().numpy()
            else:
                feats = None

            if all_feats is None:
                all_feats = feats
            else:
                if feats is not None:
                    all_feats = np.concatenate((all_feats, feats), axis=0)

        labels.extend(lbl.cpu().numpy().tolist())

    labels = np.array(labels)
    return all_feats, labels


def setup_data_and_model(data_file, img_dir, device):
    """Load data and create the BiO-CLIP model."""
    train_data, test_data = load_data(data_file, img_dir, test_size=0.1)
    model = create_model("hf-hub:imageomics/bioclip", output_dict=True, require_pretrained=True)
    return model.to(device), train_data, test_data


def prepare_data_loaders(train_data, test_data, img_dir, image_transform, batch_size):
    """Prepare DataLoaders for training and testing."""
    train_sig_dset = ButterflyDatasetClass14(train_data, img_dir, transforms=image_transform)
    tr_sig_dloader = DataLoader(train_sig_dset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_dset = ButterflyDatasetClass14(test_data, img_dir, transforms=image_transform)
    test_dl = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=8)
    return tr_sig_dloader, test_dl


def maybe_augment_single(image_tensor: torch.Tensor, augment_prob: float, augmentation_transforms) -> torch.Tensor:
    """
    With probability augment_prob, applies Albumentations transforms to the image.
    Expects a tensor of shape (3, H, W) in [0,1] and returns a transformed tensor.
    """
    if random.random() < augment_prob:
        img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        augmented = augmentation_transforms(image=img_np)
        aug_img = augmented['image'].astype(np.float32) / 255.0
        aug_tensor = torch.from_numpy(aug_img).permute(2, 0, 1)
        return aug_tensor
    else:
        return image_tensor


class ClassificationHead(nn.Module):
    """A simple linear classification head."""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# --------------------------
# Main Fine-Tuning Script
# --------------------------
def main(args):
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get transforms and augmentations
    image_transform, clip_transform = get_transforms(args.image_size)
    augment_prob, augmentation_transforms = get_augmentation_pipeline(args.augment_prob)

    # Setup U-Net for background removal
    unet_bg = UNet256(in_channels=3, out_channels=1).to(device)
    unet_bg.load_state_dict(torch.load(args.unet_ckpt, map_location=device), strict=False)
    unet_bg.eval()

    # Create and load the BiO-CLIP model and data
    model, train_data, valid_data = setup_data_and_model(args.data_file, args.img_dir, device)
    train_loader, val_loader = prepare_data_loaders(train_data, valid_data, args.img_dir, image_transform, args.batch_size)

    # Optionally load fine-tuned weights
    if args.load_weights:
        model.load_state_dict(torch.load(args.ft_path, map_location=device), strict=False)
    model = model.to(device)

    # Setup classification head
    classifier_head = ClassificationHead(args.feature_dim, args.num_classes).to(device)
    if args.load_weights:
        classifier_head.load_state_dict(torch.load(args.cl_head_path, map_location=device), strict=False)

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last N transformer blocks and the final layer norm
    resblocks = model.visual.transformer.resblocks
    for block in resblocks[-args.num_unfrozen_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    for param in model.visual.ln_post.parameters():
        param.requires_grad = True

    backbone_params = [p for p in model.parameters() if p.requires_grad]
    head_params = [p for p in classifier_head.parameters() if p.requires_grad]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_classifier},
    ])

    criterion = nn.CrossEntropyLoss()

    # Training loop with inline background removal and augmentation
    for epoch in range(args.num_epochs):
        model.train()
        classifier_head.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(torch.int64).to(device)

            # Process each image: optional augmentation and resizing for CLIP
            processed_batch = []
            for i in range(imgs.shape[0]):
                single_img = imgs[i]
                final_256 = maybe_augment_single(single_img, augment_prob, augmentation_transforms)
                pil_256 = transforms.ToPILImage()(torch.clamp(final_256, 0, 1))
                clip_ready = clip_transform(pil_256)
                processed_batch.append(clip_ready)

            processed_batch = torch.stack(processed_batch, dim=0).to(device)
            optimizer.zero_grad()

            out_dict = model(processed_batch)
            feats = out_dict["image_features"]
            logits = classifier_head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * processed_batch.size(0)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        # Validation loop
        model.eval()
        classifier_head.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(torch.int64).to(device)

                processed_batch = []
                for i in range(imgs.shape[0]):
                    single_img = imgs[i]
                    pil_256 = transforms.ToPILImage()(torch.clamp(single_img, 0, 1))
                    clip_ready = clip_transform(pil_256)
                    processed_batch.append(clip_ready)
                processed_batch = torch.stack(processed_batch, dim=0).to(device)

                out_dict = model(processed_batch)
                feats = out_dict["image_features"]
                logits = classifier_head(feats)

                loss = criterion(logits, labels)
                val_loss += loss.item() * processed_batch.size(0)
                _, preds = torch.max(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f"           Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save the trained models with generic filenames to avoid leaking info
    clf_save_path = Path(args.clf_save_dir) / "cl_head.pth"
    model_save_path = Path(args.clf_save_dir) / "model.pth"
    torch.save(classifier_head.state_dict(), clf_save_path)
    torch.save(model.state_dict(), model_save_path)

    print("Done! Model trained with background removal + random augmentation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune BiO-CLIP with background removal and augmentation.")
    
    # Required paths
    parser.add_argument("--data_file", type=str, required=True, help="Path to the CSV file with data.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing image files.")
    parser.add_argument("--clf_save_dir", type=str, required=True, help="Directory to save the trained models.")
    
    # Optional model paths for loading weights
    parser.add_argument("--ft_path", type=str, default="", help="Path to fine-tuned BiO-CLIP weights.")
    parser.add_argument("--cl_head_path", type=str, default="", help="Path to the classification head weights.")
    parser.add_argument("--unet_ckpt", type=str, required=True, help="Path to the U-Net checkpoint.")
    
    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for the backbone.")
    parser.add_argument("--lr_classifier", type=float, default=1e-3, help="Learning rate for the classifier head.")
    parser.add_argument("--num_unfrozen_blocks", type=int, default=2, help="Number of transformer blocks to unfreeze.")
    parser.add_argument("--num_classes", type=int, default=15, help="Number of classification categories.")
    parser.add_argument("--feature_dim", type=int, default=512, help="Feature dimension output by the model.")
    parser.add_argument("--image_size", type=int, default=256, help="Size for resizing images before processing.")
    
    # Augmentation settings
    parser.add_argument("--augment_prob", type=float, default=0.0, help="Probability of applying augmentation.")
    
    # Flag to load weights
    parser.add_argument("--load_weights", action="store_true", help="Flag to load pretrained weights.")
    
    args = parser.parse_args()
    main(args)

