import torch
from torch import nn

class CNNmodel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5),
            nn.ReLU(),

            nn.Conv2d(5, 10, kernel_size=3),
            nn.ReLU(),

            nn.Conv2d(10, 50, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(50*75*75, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        def forward(self, x): #x is 150x150 image
            x = self.cnn_layers(x)
            x = self.flatten(x)
            x = self.fc_layers(x)
            return x



            