import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import random
import splitfolders

TRAIN_DATASET_PATH = r"D:/AI_PROJECT/BRaTs/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
# TEST_PATH = r"D:/AI_PROJECT/BRaTs/BraTS2020_ValidationData"
INPUT_DATASET_PATH = r"D:/AI_PROJECT/BRaTs/BraTS2020_TrainingData/Data Processed"
OUPUT_DATASET_PATH = r"D:/AI_PROJECT/BRaTs/BraTS2020_TrainingData/Splitted Data"

def to_categorical(labels, num_classes):
    # Chuyển đổi mảng 3D sang 1D
    labels_flat = labels.flatten()
    # Tạo mảng one-hot cho mảng 1D
    one_hot = np.zeros((labels_flat.size, num_classes))
    one_hot[np.arange(labels_flat.size), labels_flat] = 1
    # Chuyển đổi lại thành 3D với số lớp ở trục cuối
    one_hot_3d = one_hot.reshape(*labels.shape, num_classes)
    return one_hot_3d

if __name__ == "__main__":
    scaler = MinMaxScaler()
    t2_list = sorted(glob.glob(TRAIN_DATASET_PATH + '*/*t2.nii'))
    t1ce_list = sorted(glob.glob(TRAIN_DATASET_PATH + '*/*t1ce.nii'))
    flair_list = sorted(glob.glob(TRAIN_DATASET_PATH + '*/*flair.nii'))
    mask_list = sorted(glob.glob(TRAIN_DATASET_PATH + '*/*seg.nii'))

    print(t2_list[1])
    print(t1ce_list[1])
    print(flair_list[1])
    print(mask_list[1])

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(INPUT_DATASET_PATH + "/images", exist_ok=True)
    os.makedirs(INPUT_DATASET_PATH + "/mask", exist_ok=True)

    for image in range(len(t2_list)):
        t2_image = nib.load(t2_list[image]).get_fdata()
        t1ce_image = nib.load(t1ce_list[image]).get_fdata()
        flair_image = nib.load(flair_list[image]).get_fdata()
        mask = nib.load(mask_list[image]).get_fdata()

        t2_image = scaler.fit_transform(t2_image.reshape(-1, 1)).reshape(t2_image.shape)
        t1ce_image = scaler.fit_transform(t1ce_image.reshape(-1, 1)).reshape(t1ce_image.shape)
        flair_image = scaler.fit_transform(flair_image.reshape(-1, 1)).reshape(flair_image.shape)
        mask = mask.astype(np.uint8)
        mask[mask == 4] = 3

        images_combined = np.stack([t2_image, t1ce_image, flair_image], axis=3)
        cropped_image = images_combined[56:184, 56:184, 13:141]
        cropped_mask = mask[56:184, 56:184, 13:141]

        val, counts = np.unique(mask, return_counts=True)
        if (1 - (counts[0] / counts.sum())) > 0.01:
            mask = to_categorical(mask, num_classes=4)
            np.save(INPUT_DATASET_PATH + "/images/image_" + str(image) + ".npy", cropped_image)
            np.save(INPUT_DATASET_PATH + "/mask/image_" + str(image) + ".npy", mask)

    splitfolders.ratio(INPUT_DATASET_PATH, output=OUPUT_DATASET_PATH, ratio=(.8, .2), seed=42, group_prefix=None)
