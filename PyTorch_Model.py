#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# In[2]:


#Define Transformers to Normalize the data and conver it to tensors
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])


# In[3]:


#Load the dataset
train_data = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_data = datasets.MNIST(root='./data',train=False,download=True,transform=transform)


# In[4]:


#Create Dataloaders
train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=False)


# In[21]:


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)  # 1x28x28 -> 6x28x28
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)  # 6x28x28 -> 12x28x28
        self.fc1 = nn.Linear(12 * 7 * 7, 32)  # Reduce fully connected size
        self.fc2 = nn.Linear(32, 10)  # 10 output classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)  # Downsample: 6x28x28 -> 6x14x14
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)  # Downsample: 12x14x14 -> 12x7x7
        x = x.view(-1, 12 * 7 * 7)  # Flatten for the fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# In[22]:


#create the mdodel
def get_model():
    model = MNISTModel()  # Create the model instance
    return model

model = get_model()


# In[23]:


#Loss function & Optimizers
criterion = nn.CrossEntropyLoss()


# In[24]:


optimizer = optim.Adam(model.parameters(), lr=0.001)  # Correct way


# In[25]:


#Train the Model
for epoch in range(1):
    for images, lables in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,lables)
        loss.backward()
        optimizer.step()
        
    print(f'Epoch {epoch+1},loss:{loss.item()}')
    


# In[26]:


model.eval()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def eval_model(model,test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():  # No need to calculate gradients
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100*correct/total

model_accu = eval_model(model,test_loader)


