import torch
import pytest
import torch.nn as nn
from model import Net

@pytest.fixture
def model():
    return Net()

class TestModelStructure:
    """Pre-training tests to verify model architecture and structure"""
    
    def test_parameter_count(self, model):
        """Test if model has less than 20k parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params < 20000, f"Model has {total_params} parameters, should be less than 20000"

    def test_forward_shape(self, model):
        """Verify model output shape is correct"""
        batch_size = 64
        x = torch.randn(batch_size, 1, 28, 28)
        output = model(x)
        assert output.shape == (batch_size, 10), f"Expected shape (64, 10), got {output.shape}"

    def test_model_stability(self, model):
        """Check if model produces stable outputs (no NaN values)"""
        x = torch.randn(10, 1, 28, 28)
        output = model(x)
        assert not torch.isnan(output).any(), "Model produced NaN values"

    def test_batch_normalization_presence(self, model):
        """Verify presence of batch normalization layers"""
        has_batch_norm = any(
            isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
            for module in model.modules()
        )
        assert has_batch_norm, "Model should contain batch normalization layers"

    def test_dropout_presence(self, model):
        """Verify presence of dropout layers"""
        has_dropout = any(
            isinstance(module, nn.Dropout) for module in model.modules()
        )
        assert has_dropout, "Model should contain dropout layers"

    def test_final_layer_architecture(self, model):
        """Verify model ends with either FC layer or GAP"""
        modules = list(model.modules())
        has_valid_end = False
        for module in modules[-3:]:
            if isinstance(module, (nn.Linear, nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                has_valid_end = True
                break
        assert has_valid_end, "Model should end with either FC layer or GAP"
  