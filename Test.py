import pytest
import os
import PyTorch_Model
import torch

def test_model_params():
    model = PyTorch_Model.MNISTModel()
    params_count = PyTorch_Model.count_params(model)
    assert params_count <=25000, "Model has more than 25K Params"

def test_model_accuracy():
    model = PyTorch_Model.MNISTModel()
    print("Model intialised")
    
    test_loader = PyTorch_Model.test_loader
    print("Test data loader initialised")

    # Verify the model's state dictionary (weights)
    print("Model state dict keys: ", model.state_dict().keys())

    # Verify one batch from the test loader
    for images, labels in test_loader:
        print(f"Batch size: {images.shape}, Labels: {labels[:5]}")
        break

    # Move the model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Running on device: {device}")
    
    accuracy = PyTorch_Model.eval_model(model,test_loader)
    assert accuracy >=95, "Model accuracy is below 95%"
