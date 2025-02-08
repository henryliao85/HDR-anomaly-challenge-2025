import os
import glob
import cv2
import numpy as np
import pandas as pd
import argparse

from utils import load_dataset_images

def crop_wings(test_img, predicted_img, padding=50, cropped_dim=(256, 256)):
  '''Goes through the predicted mask of an image and crops down the original image to each of the 4 available wings
  based on the predicted segmented wing mask'''

  cropped_wings = dict()
  cropped_wings_resized = dict()

  #only search for masks belonging to right/left hindwings and forewings
  for wing_class in [1,2,3,4]: #[2,3,4,5]:
    img = test_img #[:,:, 0]
    mask = np.asarray(predicted_img==wing_class, dtype=int) #0s and 1s

    y_coords = []
    x_coords = []
    for y in range(0, mask.shape[0]):
      for x in range(0, mask.shape[1]):
        if mask[y,x]==1:
          x_coords.append(x)
          y_coords.append(y)

    #our mask is empty - therefore no existing mask for that wing
    if len(x_coords) == 0 and len(y_coords) == 0:
      print("empty mask for wing key:", wing_class)
      continue

    #sort the x and y coordinates of our segmented wing mask to crop accordingly
    x_coords.sort()
    y_coords.sort()

    miny = y_coords[0]
    maxy = y_coords[-1]
    minx = x_coords[0]
    maxx= x_coords[-1]

    #get boundaries of segmented mask with some extra room
    if miny >= padding:
      miny -= padding
    
    if minx >= padding:
      minx -= padding
    
    if maxy <= (mask.shape[0] - padding):
      maxy += padding
    
    if maxx <= (mask.shape[1] - padding):
      maxx += padding

    #crop image down to segmented wings
    cropped_result = img[miny:maxy, minx:maxx, :]
    cropped_result_resized = cv2.resize(cropped_result, cropped_dim, interpolation=cv2.INTER_CUBIC) #resize to provided dimensions

    #store results in dictionary {wing_class: cropped_image}
    cropped_wings[wing_class] = cropped_result 
    cropped_wings_resized[wing_class] = cropped_result_resized #resize to provided dimensions

  #return both original sized and resized cropped wings *just in case*
  return cropped_wings, cropped_wings_resized


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Directory containing images we want to predict masks for. ex: /User/micheller/data/jiggins_256_256")
    parser.add_argument("--masks", required=True, help="Directory containing masks for images.")
    parser.add_argument("--pad", required=False, default=50, type=int, help="Padding in pixels to add around the wing when cropping out.")
    parser.add_argument("--output_folder", required=True, help="Directory where we should save the cropped wings to.")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # load in our images
    image_dataset_folder = args.images + '/*'
    dataset_images, image_filepaths = load_dataset_images(image_dataset_folder, color_option=1)

    #load in our masks
    mask_dataset_folder = args.masks + '/*'
    dataset_masks, mask_filepaths = load_dataset_images(mask_dataset_folder)
    
    #create a dataframe to store all metadata associated with predicted masks
    classes = {0: 'background',
            1: 'right_forewing',
            2: 'left_forewing',
            3: 'right_hindwing',
            4: 'left_hindwing',
            5: 'ruler',
            6: 'white_balance',
            7: 'label',
            8: 'color_card',
            9: 'body'}
    

    errors = []
    for (image, mask), fp in zip(zip(dataset_images, dataset_masks), image_filepaths):   
        #crop + extract any existing wings
        cropped_wings, cropped_wings_resized = crop_wings(image, mask, args.pad)

        #save each individual wing with a traceable name to the source image 
        #(ex. erato_0001_wing_2.png denotes the image contains the right forewing for erato_0001.png)
        for wing_idx in cropped_wings.keys():
            cropped_wing = cropped_wings[wing_idx]
            cropped_wing_resized = cropped_wings_resized[wing_idx]

            #create path to save the resized wing crops to
            new_folder = fp.replace(image_dataset_folder.replace("*", ""), args.output_folder + '/')
            ext = new_folder.split('.')[-1]
            cropped_wing_path = new_folder.replace(f'.{ext}', f'_wing_{wing_idx}.png')
            r = "/" + cropped_wing_path.split('/')[-1]
            cropped_wing_folder = cropped_wing_path.replace(r, "")
            os.makedirs(cropped_wing_folder, exist_ok=True)

            #save the cropped wings to their path (these images will be resized to cropped_dim)
            try:
                print(f"saving: {cropped_wing_path}")
                cv2.imwrite(cropped_wing_path, cv2.cvtColor(cropped_wing, cv2.COLOR_RGB2BGR))
            except FileNotFoundError:
                errors.append(cropped_wing_path)
    
    print('The following images could encountered errors during cropping/resizing:', errors)
    return


if __name__ == "__main__":
    main()