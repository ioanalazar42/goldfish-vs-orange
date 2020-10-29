''' Trains CNN to classify goldfish and oranges '''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from datareader import load_training_data
from neuralnet import NeuralNet
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter

# set hyperparameters
LEARNING_RATE = 0.0001
MINBATCH = 32
NUM_EPOCHS = 50

# save model
MODEL_PATH = 'neuralnet.pth'

# log stats
TENSORBOARD_DIR = 'tensorboard/goldfish_orange'

# Set up the network
neuralnet = NeuralNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess image
print('Loading training images...')
training_images, training_labels = load_training_data()

# Transform images to tensor and normalize them
mean = (255 - 0) / 2 # images contain values in range [0, 255]
training_images = (training_images.astype(np.float32) - mean) / mean
training_images = torch.tensor(training_images, device=device)

# Transform training labels to tensor
training_labels = torch.tensor(training_labels, device=device, dtype=torch.int64)

# Set up TensorBoard
writer = SummaryWriter(TENSORBOARD_DIR)
writer.add_graph(neuralnet, training_images)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(neuralnet.parameters(), lr=LEARNING_RATE)

print('Start training...')
start_time_training = timer()
for epoch in range(NUM_EPOCHS):
    start_time_epoch = timer()
    average_loss = 0.0

    # 100 mini-batch updates per epoch
    for i in range(100):
        random_indexes = np.random.choice(len(training_images), MINBATCH)
        minbatch = training_images[random_indexes]
        minbtach_labels = training_labels[random_indexes]

        optimizer.zero_grad()

        # Forward pass
        outputs = neuralnet(minbatch)
        loss = criterion(outputs, minbtach_labels)

        # Backprop
        loss.backward()
        optimizer.step()

        average_loss += loss.item()

    time_elapsed_epoch = timer() - start_time_epoch
    print(('Epoch: {} - Loss: {:.6f} - Time: {:.3f}s').format(epoch + 1, average_loss / MINBATCH, time_elapsed_epoch))
    writer.add_scalar('training/loss', average_loss, epoch)
    writer.add_scalar('training/epoch_duration', time_elapsed_epoch, epoch)

time_elapsed_training = timer() - start_time_training
print('Total training time: {:.3f}s'.format(time_elapsed_training))
print('Saving model as "{}"...'.format(MODEL_PATH))
torch.save(neuralnet.state_dict(), MODEL_PATH)
