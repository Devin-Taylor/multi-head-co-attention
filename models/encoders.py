import torch
import torch.nn as nn
import torch.nn.functional as F


class MethEncoder(nn.Module):
    def __init__(self, feature_size, embedding_size):
        super(MethEncoder, self).__init__()
        self.embeddings = nn.Parameter(torch.randn(feature_size, embedding_size, requires_grad=True))

    def forward(self, x):
        x = torch.unsqueeze(x, 2) # add in dimension
        x = self.embeddings * x
        return x

class SPECTEncoder(nn.Module):
    def __init__(self):
        super(SPECTEncoder, self).__init__()

        self.conv1 = nn.Conv3d(1, 16, 3, 1)
        self.maxpool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, 3, 1)
        self.maxpool2 = nn.MaxPool3d(2)
        self.do1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv3d(32, 64, 3, 1)
        self.maxpool3 = nn.MaxPool3d(2)
        self.conv4 = nn.Conv3d(64, 64, 3, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.do1(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = self.conv4(x)
        return x
