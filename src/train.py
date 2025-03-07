import numpy as np
import torch
from torch import nn
from dataset import LensDataset
from torch.utils.data import DataLoader
from model import CNNmodel
import matplotlib.pyplot as plt
#training modules
import torch.optim as optim

images_path = "../data/processed/all_train_data.npy"
labels_path = "../data/processed/all_train_labels.npy" 

val_images_path = "../data/processed/all_val_data.npy"
val_labels_path = "../data/processed/all_val_data.npy"

BATCH_SIZE = 32
NUM_EPOCHS = 5

model_save_path = "../models/model.pth"

train_losses = []
val_losses = []

def train_model():
    train_data = LensDataset(images_path, labels_path)
    val_data = LensDataset(val_images_path, val_labels_path)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNmodel(num_classes=3)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        

        for images, labels in train_loader:
            

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        #print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

        #validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

    torch.save(model.state_dict(), model_save_path)
    #Plot Training & Validation Loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

