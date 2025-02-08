#!/usr/bin/env python3
"""
This script reads image metadata from a CSV file, applies augmentation transforms
to images in order to reach a minimum number of images per class, and saves both
the original and augmented images with generic filenames. A new CSV file is produced
containing metadata for both real and synthetic images.

Usage:
    python augment.py --orig_img_folder path/to/original_images \
                      --output_img_folder path/to/output_images \
                      --csv_path path/to/input.csv \
                      --output_csv_path path/to/output.csv \
                      [--min_images_per_class 200] [--aug_per_image_high_count 1]  
"""

import os
import math
import argparse
import uuid

import pandas as pd
from PIL import Image
import cv2
import numpy as np
import albumentations as A  # Albumentations for image augmentations


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
    parser.add_argument("--min_images_per_class", type=int, default=200,
                        help="Minimum target images per class.")
    parser.add_argument("--aug_per_image_high_count", type=int, default=1,
                        help="Number of augmentations per original image when sufficient images exist.")
    return parser.parse_args()


# Define the augmentation transform pipeline using Albumentations
def get_augmentation_pipeline():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ElasticTransform(alpha=0.5, sigma=30, alpha_affine=30, p=0.2),
        A.GridDistortion(num_steps=4, distort_limit=0.1, p=0.2),
        A.OpticalDistortion(distort_limit=0.02, shift_limit=0.02, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.MedianBlur(blur_limit=3, p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.RandomGamma(gamma_limit=(90, 110), p=0.3),
    ])


def generate_generic_filename():
    """Generate a generic filename using UUID with PNG extension."""
    return f"{uuid.uuid4().hex}.png"


def assign_class(row):
    """
    Assign class label.
    If 'hybrid_stat' is 'hybrid', returns '14.0';
    otherwise, returns the string version of 'subspecies' or 'unknown' if missing.
    """
    if row['hybrid_stat'] == 'hybrid':
        return '14.0'
    else:
        return str(row['subspecies']) if pd.notna(row['subspecies']) else 'unknown'


def main(args):
    # Create output folder if it does not exist.
    if not os.path.exists(args.output_img_folder):
        os.makedirs(args.output_img_folder)

    # Read the original metadata CSV
    df_orig = pd.read_csv(args.csv_path)

    # Create a new column 'class' and filter out unknowns.
    df_orig['class'] = df_orig.apply(assign_class, axis=1)
    df_orig = df_orig[df_orig['class'] != 'unknown'].dropna(subset=['class'])

    # Build a dictionary mapping each class to a list of tuples (file_path, metadata_row)
    class_to_files = {}
    for _, row in df_orig.iterrows():
        cls = row['class']
        filename = row['filename']
        # Force a .png extension regardless of original CSV content.
        png_filename = os.path.splitext(filename)[0] + ".png"
        file_path = os.path.join(args.orig_img_folder, png_filename)
        if not os.path.isfile(file_path):
            print(f"Warning: File {file_path} not found.")
            continue
        class_to_files.setdefault(cls, []).append((file_path, row))

    # Prepare list to hold metadata for all output images.
    augmented_metadata = []

    # Process original images: save them with generic filenames.
    for _, row in df_orig.iterrows():
        new_row = row.copy()
        new_filename = generate_generic_filename()
        new_row['filename'] = new_filename
        new_row['synthetic'] = False
        augmented_metadata.append(new_row)

    # Save original images to the output folder with generic filenames.
    # (Keep a mapping to avoid re-saving if the same image appears more than once.)
    orig_save_map = {}

    # Get augmentation pipeline.
    augmentation_transforms = get_augmentation_pipeline()

    # Main augmentation loop for each class.
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
            # Save the original image with a generic name if not already done.
            if img_path not in orig_save_map:
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to read image {img_path}")
                        continue
                    # Convert from BGR to RGB.
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"Failed to open {img_path}: {e}")
                    continue

                new_orig_filename = generate_generic_filename()
                save_path = os.path.join(args.output_img_folder, new_orig_filename)
                Image.fromarray(img).save(save_path)
                orig_save_map[img_path] = new_orig_filename

            # Generate augmented images.
            try:
                # Re-read image to ensure fresh copy.
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to read image {img_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Failed to open {img_path}: {e}")
                continue

            for i in range(aug_per_image):
                augmented = augmentation_transforms(image=np.array(img))
                img_aug = augmented['image']

                new_aug_filename = generate_generic_filename()
                save_path = os.path.join(args.output_img_folder, new_aug_filename)
                Image.fromarray(img_aug).save(save_path)

                # Create metadata entry for the synthetic image.
                new_metadata = metadata_row.copy()
                new_metadata['filename'] = new_aug_filename
                new_metadata['synthetic'] = True
                augmented_metadata.append(new_metadata)

    print("Data augmentation complete.")

    # Save new metadata CSV.
    df_augmented = pd.DataFrame(augmented_metadata)
    df_augmented.to_csv(args.output_csv_path, index=False)
    print(f"Augmented metadata CSV saved to {args.output_csv_path}.")


if __name__ == '__main__':
    args = parse_args()
    main(args)

