import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        super().__init__()
        self.dataset_path = dataset_path
        # Lọc file .npy, bỏ qua file ẩn
        self.file_names = [f for f in os.listdir(dataset_path) if f.endswith(".npy") and not f.startswith(".")]
        self.transform = transform

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        file_path = os.path.join(self.dataset_path, self.file_names[index])
        data = np.load(file_path)
        data = torch.tensor(data, dtype=torch.float32) 
        if self.transform:
            data = self.transform(data)

        return data
    
def create_loader(dataset: CustomDataset):
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return loader


if __name__ == "__main__":
    train_dataset_path = "D:/AI_PROJECT/BRaTs/BraTS2020_TrainingData/Splitted Data/train/images"

    train_dataset = CustomDataset(train_dataset_path)
    # data_len = len(train_dataset)  
    # print(f"Number of files in dataset: {len(train_dataset)}")
    # print(f"Indexing 0 {train_dataset[0].shape}")

    train_loader = create_loader(train_dataset)
    print(f"Number of batches in train_loader: {len(train_loader)}")
    
    for batch in train_loader:
        print(f"Shape of one batch: {batch.shape}")
        break 

