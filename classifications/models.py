import torch
import torch.nn as nn

class StandardClassifier(nn.Module):
    def name(self):
        return 'Stanndard Classifier'

    def __init__(self, input_channels, num_classes):
        super(StandardClassifier, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Input image size: 3 x 32 x 32
        assert x.shape[-2] == 32 and x.shape[-1] == 32
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn4(x)

        x = x.view(-1, 512 * 2 * 2)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x
