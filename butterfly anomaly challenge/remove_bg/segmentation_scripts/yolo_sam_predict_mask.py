import os
import glob
import pandas as pd
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import torch, torchvision
import argparse
import wget

from utils import load_dataset_images, read_image_paths

sam_checkpoint = "/home/string-1/Documents/hdr_butterfly/HDR-anomaly-challenge-sample/segmentation/wing-segmentation/sam_vit_l_0b3195.pth"
yolo_checkpoint = "/home/string-1/Documents/hdr_butterfly/HDR-anomaly-challenge-sample/segmentation/wing-segmentation/yolo_detection_8m_shear_10.0_scale_0.5_translate_0.1_fliplr_0.0_best.pt"

def get_mask(r, image_path, predictor):
    # read in image
    image = cv2.imread(image_path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # configurate SAM model to use current image
    predictor.set_image(image)

    # get bbox info from our yolo result
    bboxes_labels = r.boxes.cls
    bboxes_xyxy = r.boxes.xyxy

    # accumulate all mask componenets into a single mask
    img_mask = np.zeros((image.shape[0], image.shape[1]))
    for idx, bbox_label in enumerate(bboxes_labels):
        input_box = np.array(bboxes_xyxy[idx].tolist())
        pred_label = bbox_label.item()
        
        mask, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        mask = mask.squeeze()
        mask = mask * pred_label
        img_mask += mask
        
        # resolve overlapping mask values
        overlapping_pixels = (mask != 0) & (img_mask != 0)
        img_mask[overlapping_pixels] = mask[overlapping_pixels]
        
    img_mask = img_mask.astype(np.uint8)
    return img_mask


def get_sam_model(device):
    '''Get the SAM VIT l Model'''
    model_type = "vit_l"
    
    #download model file is not already downloaded
    if not os.path.exists(sam_checkpoint):
        # download from huggingface
        model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
        sam_checkpoint = wget.download(model_url)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)

def get_yolo_model():
    '''Download trained yolo v8 model from huggingface and load in weights'''

    ## check if we already have the trained yolo checkpoint file
    if os.path.exists(yolo_checkpoint):
        checkpoint = yolo_checkpoint
    else:
        # download from huggingface
        model_url = "https://huggingface.co/imageomics/butterfly_detection_yolo/resolve/main/yolo_detection_8m_shear_10.0_scale_0.5_translate_0.1_fliplr_0.0_best.pt"
        checkpoint = wget.download(model_url)

    model = YOLO(checkpoint)
    return model

def batch_indices_np(total, batch_size):
    indices = np.arange(total)
    return np.array_split(indices, np.ceil(total / batch_size))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Directory containing images we want to predict masks for. ex: /User/micheller/data/jiggins_256_256")
    parser.add_argument("--mask_csv", required=False, default = 'dataset_segmentation_info.csv', help="Path to the csv created containing \
                        which segmentation classes are present in each image's predicted mask.")
    return parser.parse_args()

def main():
    args = parse_args()

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
    
    # Leverage GPU if available
    use_cuda = torch.cuda.is_available()
    DEVICE   = torch.device("cuda:0" if use_cuda else "cpu")
    print("Device: ", DEVICE)

    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
    
    # read images in
    dataset_folder = args.dataset + '/*'
    image_filepaths = read_image_paths(dataset_folder) #just read in image filepaths since model handles converting path to image
    print(f'Number of images in dataset: {len(image_filepaths)}')

    dataset_segmented = pd.DataFrame(columns = ['image', 'background', 
                                            'right_forewing', 'left_forewing', 
                                            'right_hindwing', 'left_hindwing', 
                                            'ruler', 'white_balance', 
                                            'label', 'color_card', 'body'])
    

    # load trained yolo detection model
    bbox_model = get_yolo_model()

    # load SAM model
    segmentation_model = get_sam_model(DEVICE)

    i = 0 #df indexer
    batches = batch_indices_np(len(image_filepaths), 16)
    for batch in batches: 
        print('Current batch:', batch)
        image_fp_batch = [image_filepaths[i] for i in batch]

        # predict bboxes for all components in our images 
        results = bbox_model.predict(image_fp_batch, verbose=False)
    
        # create masks using SAM (segment-anything-model)
        print("Using YOLO BBOX to Get SAM Mask...")
        for r, fp in zip(results, image_fp_batch):
            #get the mask with category id's as pixel values
            mask = get_mask(r, fp, segmentation_model) 
            
            #create the path to which the mask will be saved, replicating the folder + naming structure of the input dataset
            mask_path = fp.replace(args.dataset, f'{args.dataset}_masks')
            mask_path = mask_path.replace(f".{fp.split('.')[-1]}", "_mask.png") #replace extension and save mask as a png
            
            #create the folder in which the mask will be saved in if it doesn't exist already
            mask_filename = "/" + mask_path.split('/')[-1] #
            mask_folder = mask_path.replace(mask_filename, "")
            os.makedirs(mask_folder, exist_ok=True)
            
            #save mask with cv2 to preserve pixel categories
            print(f"Mask path:{mask_path}")
            cv2.imwrite(mask_path, mask)

            #enter relevant segmentation data for the image in our dataframe
            classes_in_image = np.unique(mask)
            print('Classes in image:', classes_in_image)
            classes_not_in_image = set(classes.keys()) ^ set(classes_in_image)
            dataset_segmented.loc[i, 'image'] = fp
            
            #enter `1` for all segmentation classes that appear in our mask
            for val in classes_in_image:
                pred_class = classes[val]
                dataset_segmented.loc[i, pred_class] = 1 #class exists in segmentation mask

            #enter `0` for all segmentation classes that were not predicted
            for not_pred_val in classes_not_in_image:
                not_pred_class = classes[not_pred_val]
                dataset_segmented.loc[i, not_pred_class] = 0 #class does not exist in segmentation mask

            i += 1

    # Save csv containing information about segmentation masks per each image
    dataset_segmented.to_csv(args.segmentation_csv, index=False)

    return

if __name__ == "__main__":
    main()
