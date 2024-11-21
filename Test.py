def test_model():
    num_params = count_params(model)
    assert num_params >=25000, f"Model has too many params: {num_params} (> 25000)"
    
    accuracy = eval_model(model, test_loader)
    assert accuracy >=95.0, f"Model accuracy is low: {accuracy} (< 95%)"
