import pytest
import os
import PyTorch_Model
import torch

@pytest.fixture
def model():
    model = PyTorch_Model.MNISTModel()  # Assuming the model class is in PyTorch_Model
    return model

@pytest.fixture
def test_loader():
    # You can define the DataLoader here for testing, similar to your main code
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
    return test_loader

def test_readme_exists():
    assert os.path.isfile("README.md"), "README.md file missing!"

def test_readme_contents():
    readme_words=[word for line in open('README.md', 'r', encoding="utf-8") for word in line.split()]
    assert len(readme_words) >= 500, "Make your README.md file interesting! Add atleast 500 words"

def test_model_params():
    params_count = PyTorch_Model.count_params(model)
    assert params_count <=25000, "Model has more than 25K Params"

def test_model_accuracy():
    accuracy = PyTorch_Model.eval_model(model,test_loader)
    assert accuracy >=95, "Model accuracy is below 95%"
