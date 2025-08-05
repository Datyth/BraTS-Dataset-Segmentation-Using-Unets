import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler

import random

# TRAIN_PATH = "D:/AI_PROJECT\BRaTs\BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/" 

def visualize_folders(dataset_path, num_subfolder, n_slice = 50):
    scaler = MinMaxScaler()
    TRAIN_DATASET_PATH = dataset_path
    num_subfolder =  str(num_subfolder)
    subfolder =  dataset_path + "BraTS20_Training_" + num_subfolder 

    test_image_flair = nib.load(subfolder + "/BraTS20_Training_" + num_subfolder + "_flair.nii").get_fdata()
    test_image_t1 = nib.load(subfolder + "/BraTS20_Training_" + num_subfolder + "_t1.nii").get_fdata()
    test_image_t1ce = nib.load(subfolder + "/BraTS20_Training_" + num_subfolder + "_t1ce.nii").get_fdata()
    test_image_t2 = nib.load(subfolder + "/BraTS20_Training_" + num_subfolder + "_t2.nii").get_fdata()

    test_image_flair = scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(test_image_flair.shape)
    test_image_t1 = scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)
    test_image_t1ce = scaler.fit_transform(test_image_t1ce.reshape(-1, test_image_t1ce.shape[-1])).reshape(test_image_t1ce.shape)
    test_image_t2 = scaler.fit_transform(test_image_t2.reshape(-1, test_image_t2.shape[-1])).reshape(test_image_t2.shape)

    test_mask = nib.load(subfolder + "/BraTS20_Training_" + num_subfolder + "_seg.nii").get_fdata()
    test_mask = test_mask.astype(np.uint8)
    test_mask[test_mask == 4] = 3

    if(n_slice == None):
        n_slice=random.randint(0, test_mask.shape[2])

    print(f"n_slice: {n_slice}")

    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.imshow(test_image_flair[:,:,n_slice], cmap='gray')
    plt.title('Image flair')
    plt.subplot(232)
    plt.imshow(test_image_t1[:,:,n_slice], cmap='gray')
    plt.title('Image t1')
    plt.subplot(233)
    plt.imshow(test_image_t1ce[:,:,n_slice], cmap='gray')
    plt.title('Image t1ce')
    plt.subplot(234)
    plt.imshow(test_image_t2[:,:,n_slice], cmap='gray')
    plt.title('Image t2')
    plt.subplot(235)
    plt.imshow(test_mask[:,:,n_slice])
    plt.title('Mask')
    plt.show()
    


if __name__ == "__main__":
    TRAIN_DATASET_PATH = "D:/AI_PROJECT\BRaTs\BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
    #VALIDATION_DATASET_PATH = 'BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    # visualize_folders(TRAIN_DATASET_PATH, 100)
    combined_x=np.load("D:/AI_PROJECT/BRaTs/BraTS2020_TrainingData/Splitted Data/train/images/image_8.npy")
    test_mask = np.load("D:/AI_PROJECT/BRaTs/BraTS2020_TrainingData/Splitted Data/train/mask/image_8.npy")
    print(combined_x.shape)
    print(test_mask.shape)

    # n_slice=random.randint(0, test_mask.shape[2])
    n_slice = 80
    plt.figure(figsize=(16, 12))

    plt.subplot(331)
    plt.imshow(combined_x[:,:,n_slice, 0], cmap='gray')
    plt.title('Image flair')
    plt.subplot(332)
    plt.imshow(combined_x[:,:,n_slice, 1], cmap='gray')
    plt.title('Image t1ce')
    plt.subplot(333)
    plt.imshow(combined_x[:,:,n_slice, 2], cmap='gray')
    plt.title('Image t2')
    plt.subplot(334)
    plt.imshow(test_mask[:,:,n_slice, 0])
    plt.title('Mask 0')
    plt.subplot(335)
    plt.imshow(test_mask[:,:,n_slice, 1])
    plt.title('Mask 1')
    plt.subplot(336)
    plt.imshow(test_mask[:,:,n_slice, 2])
    plt.title('Mask 2')
    plt.subplot(337)
    plt.imshow(test_mask[:,:,n_slice, 3])
    plt.title('Mask 3')
    plt.show()

    print(type(combined_x[0, 0, 0, 0]))