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
    test_loader = PyTorch_Model.test_loader
    accuracy = PyTorch_Model.eval_model(model,test_loader)
    assert accuracy >=95, "Model accuracy is below 95%"
