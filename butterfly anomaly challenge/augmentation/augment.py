#!/usr/bin/env python3
"""
This script reads image metadata from a CSV file and applies torchvision-based augmentations
to generate synthetic images. Both original and augmented images are saved with generic filenames,
so no sensitive or descriptive information is exposed. A new CSV file is produced containing metadata
for both real and synthetic images.

Usage:
    python augment.py --orig_img_folder path/to/orig_images \
                      --output_img_folder path/to/output_images \
                      --csv_path path/to/input.csv \
                      --output_csv_path path/to/output.csv \
                      [--min_images_per_class 100] \
                      [--aug_per_image_high_count 3]
"""

import os
import math
import argparse
import uuid

import pandas as pd
from PIL import Image
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment images to reach a minimum count per class and output new metadata."
    )
    parser.add_argument("--orig_img_folder", type=str, required=True,
                        help="Folder with original images (PNG format).")
    parser.add_argument("--output_img_folder", type=str, required=True,
                        help="Folder to save output images.")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV metadata file.")
    parser.add_argument("--output_csv_path", type=str, required=True,
                        help="Path to save the new metadata CSV.")
    parser.add_argument("--min_images_per_class", type=int, default=100,
                        help="Minimum target images per class.")
    parser.add_argument("--aug_per_image_high_count", type=int, default=3,
                        help="Number of augmentations per original image when enough originals exist.")
    return parser.parse_args()


def get_augmentation_pipeline():
    return transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
    ])


def generate_generic_filename():
    """Generate a generic filename using a UUID with a PNG extension."""
    return f"{uuid.uuid4().hex}.png"


def assign_class(row):
    """
    Assign a class label.
    If 'hybrid_stat' is 'hybrid', returns 'hybrid';
    otherwise, returns the string version of 'subspecies' or 'unknown' if missing.
    """
    if row['hybrid_stat'] == 'hybrid':
        return 'hybrid'
    else:
        return str(row['subspecies']) if pd.notna(row['subspecies']) else 'unknown'


def main(args):
    # Create output folder if it does not exist.
    if not os.path.exists(args.output_img_folder):
        os.makedirs(args.output_img_folder)

    # Read the original metadata CSV.
    df_orig = pd.read_csv(args.csv_path)

    # Create a new column 'class' and remove rows with unknown class.
    df_orig['class'] = df_orig.apply(assign_class, axis=1)
    df_orig = df_orig[df_orig['class'] != 'unknown'].dropna(subset=['class'])

    # Build a dictionary mapping each class to a list of (file_path, metadata_row).
    class_to_files = {}
    for _, row in df_orig.iterrows():
        cls = row['class']
        filename = row['filename']
        # Force .png extension regardless of what the CSV indicates.
        png_filename = os.path.splitext(filename)[0] + ".png"
        file_path = os.path.join(args.orig_img_folder, png_filename)
        if not os.path.isfile(file_path):
            print(f"Warning: File {file_path} not found.")
            continue
        class_to_files.setdefault(cls, []).append((file_path, row))

    # Prepare a list to store metadata for all images.
    augmented_metadata = []

    # Save original images with generic filenames and add their metadata.
    # Use a mapping to avoid re-saving the same file more than once.
    orig_save_map = {}
    for _, row in df_orig.iterrows():
        orig_filename = os.path.splitext(row['filename'])[0] + ".png"
        file_path = os.path.join(args.orig_img_folder, orig_filename)
        if not os.path.isfile(file_path):
            continue
        if file_path not in orig_save_map:
            new_name = generate_generic_filename()
            try:
                img = Image.open(file_path).convert('RGB')
                img.save(os.path.join(args.output_img_folder, new_name))
                orig_save_map[file_path] = new_name
            except Exception as e:
                print(f"Failed to save original image {file_path}: {e}")
                continue
        new_row = row.copy()
        new_row['synthetic'] = False
        new_row['filename'] = orig_save_map[file_path]
        augmented_metadata.append(new_row)

    # Get the augmentation pipeline.
    augmentation_transforms = get_augmentation_pipeline()

    # Main augmentation loop per class.
    for cls, file_list in class_to_files.items():
        num_orig = len(file_list)
        print(f"Processing class '{cls}' with {num_orig} original images.")

        # Determine augmentations per original image.
        if num_orig < args.min_images_per_class:
            aug_per_image = math.ceil((args.min_images_per_class - num_orig) / num_orig)
        else:
            aug_per_image = args.aug_per_image_high_count

        print(f"--> Generating {aug_per_image} augmented images per original image.")

        for (img_path, metadata_row) in file_list:
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Failed to open {img_path}: {e}")
                continue

            # Generate augmented images.
            for i in range(aug_per_image):
                try:
                    img_aug = augmentation_transforms(img)
                except Exception as e:
                    print(f"Augmentation failed for {img_path}: {e}")
                    continue

                new_aug_filename = generate_generic_filename()
                save_path = os.path.join(args.output_img_folder, new_aug_filename)
                try:
                    img_aug.save(save_path)
                except Exception as e:
                    print(f"Failed to save augmented image {save_path}: {e}")
                    continue

                new_metadata = metadata_row.copy()
                new_metadata['filename'] = new_aug_filename
                new_metadata['synthetic'] = True
                augmented_metadata.append(new_metadata)

    print("Data augmentation complete.")

    # Save the new metadata as a CSV.
    df_augmented = pd.DataFrame(augmented_metadata)
    df_augmented.to_csv(args.output_csv_path, index=False)
    print(f"Augmented metadata CSV saved to {args.output_csv_path}.")


if __name__ == '__main__':
    args = parse_args()
    main(args)

