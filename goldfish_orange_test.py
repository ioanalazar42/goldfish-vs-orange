''' Test network trained by goldfish_orange_train '''

import numpy as np
import torch

from datareader import load_test_data
from neuralnet import NeuralNet
from timeit import default_timer as timer

MODEL_PATH = 'neuralnet.pth'

neuralnet = NeuralNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
neuralnet.to(device)
neuralnet.load_state_dict(torch.load(MODEL_PATH))

# Load and preprocess image
test_images, test_labels = load_test_data()

# Transform images to tensor and normalize them
mean = (255 - 0) / 2 # images contain values in range [0, 255]
test_images = (test_images.astype(np.float32) - mean) / mean
test_images = torch.tensor(test_images, device=device)

# Transform test labels to tensor
test_labels = torch.tensor(test_labels, device=device, dtype=torch.int64)

with torch.no_grad():
    outputs = neuralnet(test_images)
    _, predicted = torch.max(outputs.data, 1)

    correct = (predicted == test_labels).sum().item()

    print('Accuracy of the network on {} test images: {}%'.format(len(test_images), 100 * correct / len(test_images)))
    print()
    print('Predictions:')
    print(predicted)
