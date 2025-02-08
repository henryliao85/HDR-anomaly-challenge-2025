import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import argparse

from utils import load_dataset_images

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dataset_path", required=True, help="Directory containing images we have masks for. ex: /User/micheller/data/jiggins_256_256")
    parser.add_argument("--mask_dataset_path", required=True, help="Directory containing masks for images. ex: /User/micheller/data/jiggins_256_256_masks")
    parser.add_argument("--main_folder_name", required=True, help="JUST the main FOLDER NAME containing all subfolders/images. ex: jiggins_256_256")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # load in our images
    image_dataset_folder = args.image_dataset_path + '/*'
    dataset_images, image_filepaths = load_dataset_images(image_dataset_folder, 1)

    #load in our masks
    mask_dataset_folder = args.mask_dataset_path + '/*'
    dataset_masks, mask_filepaths = load_dataset_images(mask_dataset_folder)
    
    errors = []
    for (image, mask), fp in zip(zip(dataset_images, dataset_masks), image_filepaths): 
        #remove background from color image
        img_removed_background = cv2.bitwise_and(image, image, mask=mask)

        #save the image with the removed background under its own folder
        bck_img_path = fp.replace(args.main_folder_name, f'{args.main_folder_name}_removed_background')
        fn = "/" + bck_img_path.split('/')[-1]
        background_folder = bck_img_path.replace(fn, "")
        os.makedirs(background_folder, exist_ok=True)

        #save the resized cropped wings to their path
        try:
            cv2.imwrite(bck_img_path, cv2.cvtColor(img_removed_background, cv2.COLOR_RGB2BGR))
        except FileNotFoundError:
            errors.append(bck_img_path)
            
    print('The following images could encountered errors during background removal:', errors)
    return


if __name__ == "__main__":
    main()