#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms


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
test_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)


# In[21]:


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)  # Convolutional layer
        self.pool = nn.MaxPool2d(2, 2)               # Max pooling
        self.fc1 = nn.Linear(3 * 13 * 13, 32)       # Fully connected layer
        self.fc2 = nn.Linear(32, 10)                 # Output layer

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Apply ReLU
        x = self.pool(x)               # Apply pooling
        x = x.view(-1, 3 * 13 * 13)   # Flatten
        x = torch.relu(self.fc1(x))    # Fully connected layer
        x = self.fc2(x)                # Output
        
        return x


# In[22]:


#create the mdodel
model = MNISTModel()


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

def test_model():
    num_params = count_params(model)
    assert num_params <=25000, f"Model has too many params: {num_params} (> 25000)"
    
    accuracy = eval_model(model, test_loader)
    assert accuracy >=95.0, f"Model accuracy is low: {accuracy} (< 95%)"
