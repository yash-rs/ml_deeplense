import numpy as np
import torch
from torch import nn
from dataset import LensDataset
from torch.utils.data import DataLoader
from model import CNNmodel
#training modules
import torch.optim as optim

images_path = "../data/processed/all_train_data.npy"
labels_path = "../data/processed/all_train_labels.npy" 
BATCH_SIZE = 32
NUM_EPOCHS = 5

def train_model():
    train_data = LensDataset(images_path, labels_path)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    model = CNNmodel(num_classes=3)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")


