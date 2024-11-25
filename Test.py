import pytest
import os
import PyTorch_Model

def test_readme_exists():
    assert os.path.isfile("README.md"), "README.md file missing!"

def test_readme_contents():
    readme_words=[word for line in open('README.md', 'r', encoding="utf-8") for word in line.split()]
    assert len(readme_words) >= 500, "Make your README.md file interesting! Add atleast 500 words"

def test_model_params():
    PyTorch_Model.count_params(model)

def test_model_accuracy():
    PyTorch_Model.eval_model(model,test_loader)
