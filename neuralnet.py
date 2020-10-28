''' Creates a convolutional neural network '''

import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (6, 6), 2) # input: 64x64x3 picture; output: 32 activation maps of size 30x30
        self.conv2 = nn.Conv2d(32, 64, (6, 6), 3) # input: 30x30x32; output: 64 activation maps of size 9x9
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 2) # input: 9x9x64; output: 64 activation maps of size 4x4
        self.fc1 = nn.Linear(4 * 4 * 64, 128)
        self.fc2 = nn.Linear(128, 10) # fc1, fc2 are fully connected layers

    def forward(self, x):
        x = x.view(-1, 3, 64, 64) # transform Nx64x64x3 > Nx3x64x64
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 4 * 4 * 64) # transform Nx4x4x64 > Nx1024
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
