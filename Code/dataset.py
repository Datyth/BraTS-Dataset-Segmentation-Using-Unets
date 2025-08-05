import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BraSTDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        super().__init__()
        self.images_path = images_path
        self.masks_path = masks_path

        self.image_files = [f for f in os.listdir(images_path) if f.endswith(".npy") and not f.startswith(".")]
        self.mask_files = [f for f in os.listdir(masks_path) if f.endswith(".npy") and not f.startswith(".")]
        assert len(self.image_files) == len(self.mask_files), "Number of files and masks do not match!"
        self.image_files.sort() 
        self.mask_files.sort()

        self.transform = transform

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        # Load image
        image_path = os.path.join(self.images_path, self.image_files[index])
        image = np.load(image_path)  # Shape: (128, 128, 128, 3)
        
        # Permute to [channels, depth, height, width] for PyTorch
        image = torch.tensor(image, dtype=torch.float32).permute(3, 2, 0, 1)
        
        # Load mask
        mask_path = os.path.join(self.masks_path, self.mask_files[index])
        mask = np.load(mask_path)  # Shape: (128, 128, 128, 3) - đã được tiền xử lý
        
        # Chuyển từ one-hot encoding sang class indices
        # mask có shape (128, 128, 128, 3) với 3 classes
        mask_indices = np.argmax(mask, axis=-1)  # Shape: (128, 128, 128)
        
        # Permute to [depth, height, width] để match với image format
        mask_indices = mask_indices.transpose(2, 0, 1)  # [depth, height, width]
        
        mask = torch.tensor(mask_indices, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, mask

    
def create_loader(dataset: BraSTDataset, batch_size = 16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


if __name__ == "__main__":
    images_path = "D:/AI_PROJECT/BRaTs/BraTS2020_TrainingData/Splitted Data/train/images"
    masks_path = "D:/AI_PROJECT/BRaTs/BraTS2020_TrainingData/Splitted Data/train/mask"

    train_dataset = BraSTDataset(images_path, masks_path)
    train_loader = create_loader(train_dataset, batch_size=2)

    print(f"Number of batches in train_loader: {len(train_loader)}")
    
    for batch_images, batch_masks in train_loader:
        print(f"Shape of batch_images: {batch_images.shape}")
        print(f"Shape of batch_masks: {batch_masks.shape}")
        print(f"Batch_masks dtype: {batch_masks.dtype}")
        print(f"Unique values in batch_masks: {torch.unique(batch_masks)}")
        print(f"Min value: {batch_masks.min()}, Max value: {batch_masks.max()}")
        break
