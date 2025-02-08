import os
import glob
import cv2
import numpy as np

def load_dataset_images(dataset_path, color_option=0):
    '''Load in actual images from filepaths from all subfolders in the provided dataset_path'''

    file_extensions = ["jpg", "JPG", "jpeg", "png"]

    #Get training images and mask paths then sort
    image_filepaths = []
    for directory_path in glob.glob(dataset_path):
        if os.path.isfile(directory_path):
            image_filepaths.append(directory_path)
        elif os.path.isdir(directory_path):
            for ext in file_extensions:
                for img_path in glob.glob(os.path.join(directory_path, f"*.{ext}")):
                    image_filepaths.append(img_path)


    #sort image and mask fps to ensure we have the same order to index
    image_filepaths.sort()

    #get actual masks and images
    dataset_images = []

    for img_path in image_filepaths:
        if color_option == 0:
            #read image in grayscale
            img = cv2.imread(img_path, color_option)
        elif color_option == 1:
            #read in color and reverse order to RGB since opencv reads in BGR
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dataset_images.append(img)

    #Enable dataset loading for inhomogeneous data
    try:
        # For improved performance and memory usage, try to convert the list of images to a numpy array
        dataset_images = np.array(dataset_images)
    except ValueError as e:
        # But data may be inhomogeneous depending on preprocessing
        print(f"Error converting images to numpy array: {e}")
        print("Returning list of images instead.")
        return dataset_images, image_filepaths
    
    return dataset_images, image_filepaths


def read_image_paths(dataset_path):
    '''Read in all filepaths to images'''

    file_extensions = ["jpg", "JPG", "jpeg", "png"]

    #Get training images and mask paths then sort
    image_filepaths = []
    for directory_path in glob.glob(dataset_path):
        if os.path.isfile(directory_path):
            image_filepaths.append(directory_path)
        elif os.path.isdir(directory_path):
            for ext in file_extensions:
                for img_path in glob.glob(os.path.join(directory_path, f"*.{ext}")):
                    image_filepaths.append(img_path)


    #sort image and mask fps to ensure we have the same order to index
    image_filepaths.sort()
    return image_filepaths


def load_batch_images(image_filepaths, color_option=0):
    '''Load in actual images from filepaths provided'''

    #get actual masks and images
    dataset_images = []

    for img_path in image_filepaths:
        if color_option == 0:
            #read image in grayscale
            img = cv2.imread(img_path, color_option)
        elif color_option == 1:
            #read in color and reverse order to RGB since opencv reads in BGR
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dataset_images.append(img)

    #Enable dataset loading for inhomogeneous data
    try:
        # For improved performance and memory usage, try to convert the list of images to a numpy array
        dataset_images = np.array(dataset_images)
    except ValueError as e:
        # But data may be inhomogeneous depending on preprocessing
        print(f"Error converting images to numpy array: {e}")
        print("Returning list of images instead.")
        return dataset_images
    
    return dataset_images