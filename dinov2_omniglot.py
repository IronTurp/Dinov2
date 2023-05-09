# -*- coding: utf-8 -*-
"""
Created on Tue May  9 08:39:48 2023

@author: jeanfrancois.turpin
"""

# Import required libraries
import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets import Omniglot
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

#Load the dataset
dataset = Omniglot(
    root="./omniglot", download=True, transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(), 
        transforms.Resize((98, 98))
    ]),
)

# Create a DataLoader for the dataset
dataloader = DataLoader(
  dataset, shuffle=True, batch_size=64
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained DINO V2 model with ViT-S/14 architecture
def pick_dinov2_version(version):
    if version=='vits14':
        return torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
    elif version=='vitl14':
        return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda()
    else:
        return torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
        
#dinov2 = pick_dinov2_version('vits14')
dinov2 = pick_dinov2_version('vitl14')

# Move the model to the appropriate device
dinov2 = dinov2.to(device)

# Initialize empty lists for storing embeddings and targets
all_embeddings, all_targets = [], []

# Extract embeddings for all images in the dataset
with torch.no_grad():
    for images, targets in tqdm(dataloader):
        images = images.to(device)
        embedding = dinov2(images)
        all_embeddings.append(embedding)
        all_targets.append(targets)

# Concatenate all embeddings and targets into single tensors
all_embeddings = torch.cat(all_embeddings, dim=0)
all_targets = torch.cat(all_targets, dim=0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    all_embeddings.cpu().numpy(), 
    all_targets.cpu().numpy(), 
    test_size=0.3, 
    random_state=42,
)

# Train a logistic regression model on the embeddings and labels
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the performance of the logistic regression model on the test set
test_acc = model.score(X_test, y_test)
print(f'Test accuracy: {test_acc}')


