import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch import optim
from dataset import *
from model import UNet3D
import torch
import numpy as np
import matplotlib.pyplot as plt



class Trainer():
    def __init__(self, train_loader, valid_loader, model, device, num_epochs=100, lr=1e-5, optim=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.optim = optim if optim else torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = self.combined_loss  
        self.train_metrics = {"loss": [], "accuracy": [], "iou": [], "dice": []}
        self.valid_metrics = {"loss": [], "accuracy": [], "iou": [], "dice": []}

    def dice_loss(self, predictions, labels):
        """
        Comute Dice Loss.
        Args:
            predictions (torch.Tensor): model predict, shape [batch_size, num_classes, depth, height, width].
            labels (torch.Tensor): Growth Truth, shape [batch_size, depth, height, width].
        Returns:
            torch.Tensor: Dice Loss.
        """
        smooth = 1e-6
        predictions = torch.softmax(predictions, dim=1)  
        labels_one_hot = F.one_hot(labels, num_classes=predictions.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        intersection = (predictions * labels_one_hot).sum(dim=(2, 3, 4))
        union = predictions.sum(dim=(2, 3, 4)) + labels_one_hot.sum(dim=(2, 3, 4))
        dice_score = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice_score.mean()

    def combined_loss(self, predictions, labels):
        """
        Combine CrossEntropyLoss và Dice Loss.
        """
        ce_loss = nn.CrossEntropyLoss()(predictions, labels)
        dice_loss = self.dice_loss(predictions, labels)
        return ce_loss + dice_loss

    def compute_metrics(self, predictions, labels):
        """
        Compute Accuracy, IoU, and Dice Coefficient.
        """
        predicted_classes = torch.argmax(predictions, dim=1) 
        correct_pixels = (predicted_classes == labels).sum().item()
        total_pixels = labels.numel()
        accuracy = correct_pixels / total_pixels

        smooth = 1e-6
        intersection = (predicted_classes & labels).sum().item()
        union = (predicted_classes | labels).sum().item()
        iou = (intersection + smooth) / (union + smooth)

        dice_coeff = 2 * intersection / (predicted_classes.sum().item() + labels.sum().item() + smooth)

        return accuracy, iou, dice_coeff

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            train_accuracy = 0
            train_iou = 0
            train_dice = 0

            for batch_images, batch_masks in self.train_loader:
                batch_images = batch_images.to(self.device)
                batch_masks = batch_masks.to(self.device).long()

                self.optim.zero_grad()
                outputs = self.model(batch_images)

                loss = self.criterion(outputs, batch_masks)
                loss.backward()
                self.optim.step()

                train_loss += loss.item()
                accuracy, iou, dice_coeff = self.compute_metrics(outputs, batch_masks)
                train_accuracy += accuracy
                train_iou += iou
                train_dice += dice_coeff


            train_loss /= len(self.train_loader)
            train_accuracy /= len(self.train_loader)
            train_iou /= len(self.train_loader)
            train_dice /= len(self.train_loader)

            self.train_metrics["loss"].append(train_loss)
            self.train_metrics["accuracy"].append(train_accuracy)
            self.train_metrics["iou"].append(train_iou)
            self.train_metrics["dice"].append(train_dice)

            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")

            self.validate(epoch)


    def validate(self, epoch):
        self.model.eval()
        valid_loss = 0
        valid_accuracy = 0
        valid_iou = 0
        valid_dice = 0

        with torch.no_grad():
            for batch_images, batch_masks in self.valid_loader:
                batch_images = batch_images.to(self.device)
                batch_masks = batch_masks.to(self.device).long()

                outputs = self.model(batch_images)

                # Resize labels nếu kích thước không khớp
                if batch_masks.shape[1:] != outputs.shape[2:]:
                    batch_masks = F.interpolate(batch_masks.unsqueeze(1).float(), size=outputs.shape[2:], mode='nearest').squeeze(1).long()

                loss = self.criterion(outputs, batch_masks)
                valid_loss += loss.item()
                accuracy, iou, dice_coeff = self.compute_metrics(outputs, batch_masks)
                valid_accuracy += accuracy
                valid_iou += iou
                valid_dice += dice_coeff


        valid_loss /= len(self.valid_loader)
        valid_accuracy /= len(self.valid_loader)
        valid_iou /= len(self.valid_loader)
        valid_dice /= len(self.valid_loader)

        self.valid_metrics["loss"].append(valid_loss)
        self.valid_metrics["accuracy"].append(valid_accuracy)
        self.valid_metrics["iou"].append(valid_iou)
        self.valid_metrics["dice"].append(valid_dice)

        print(f"Epoch [{epoch+1}/{self.num_epochs}], Validation Loss: {valid_loss:.4f}, Accuracy: {valid_accuracy:.4f}, IoU: {valid_iou:.4f}, Dice: {valid_dice:.4f}")


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def plot_metrics(self):
        """
        Vẽ đồ thị Loss, Accuracy, IoU, và Dice Coefficient.
        """
        epochs = range(1, self.num_epochs + 1)
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.train_metrics["loss"], label="Train Loss")
        plt.plot(epochs, self.valid_metrics["loss"], label="Valid Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.train_metrics["accuracy"], label="Train Accuracy")
        plt.plot(epochs, self.valid_metrics["accuracy"], label="Valid Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.train_metrics["iou"], label="Train IoU")
        plt.plot(epochs, self.valid_metrics["iou"], label="Valid IoU")
        plt.xlabel("Epochs")
        plt.ylabel("IoU")
        plt.title("IoU")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.train_metrics["dice"], label="Train Dice")
        plt.plot(epochs, self.valid_metrics["dice"], label="Valid Dice")
        plt.xlabel("Epochs")
        plt.ylabel("Dice Coefficient")
        plt.title("Dice Coefficient")
        plt.legend()

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    train_images_path = "D:/AI_PROJECT/BRaTs/BraTS2020_TrainingData/Splitted Data/train/images"
    train_masks_path = "D:/AI_PROJECT/BRaTs/BraTS2020_TrainingData/Splitted Data/train/mask"
    valid_images_path = "D:/AI_PROJECT/BRaTs/BraTS2020_TrainingData/Splitted Data/val/images"
    valid_masks_path = "D:/AI_PROJECT/BRaTs\BraTS2020_TrainingData/Splitted Data/val/mask"

    train_dataset = BraSTDataset(train_images_path, train_masks_path)
    train_loader = create_loader(train_dataset, batch_size = 2)

    valid_dataset = BraSTDataset(valid_images_path, valid_masks_path)
    valid_loader = create_loader(valid_dataset, batch_size = 2)


    model = UNet3D(in_channels=3, out_channels=4)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(train_loader, valid_loader, model, device, num_epochs=10, lr=1e-4)

    trainer.train()
    trainer.save_model("unet3d_model.pth")
