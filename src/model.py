import torch
from torch import nn

class CNNmodel(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNmodel, self).__init__()
        self.flatten = nn.Flatten()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128*18*18, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x): #x is 150x150 image
        x = self.cnn_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x