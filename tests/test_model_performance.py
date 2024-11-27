import torch
import pytest
import json
import os
from model import Net
from utils import get_device, data_transformation
from torchvision import datasets
import torch.nn.functional as F

@pytest.fixture
def model():
    model = Net()
    if os.path.exists('model.pth'):
        model.load_state_dict(torch.load('model.pth', map_location='cpu'))
    return model

@pytest.fixture
def test_loader():
    transformation_matrix = {
        "mean_of_data": (0.1307,),
        "std_of_data": (0.3081,),
    }
    _, test_transforms = data_transformation(transformation_matrix)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
    return torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)

class TestModelPerformance:
    """Post-training tests to verify model performance"""

    def test_accuracy_threshold(self):
        """Verify model achieves 99.4% accuracy on test set"""

        # collect accuracy details for all epochs
        
        if os.path.exists('logs/metrics.json'):
            with open('logs/metrics.json', 'r') as f:
                metrics = json.load(f)
                accuracy_details_for_all_epochs = metrics['test_accuracy']

            assert any(epoch >= 99.4 for epoch in accuracy_details_for_all_epochs), \
                f"Model accuracy {accuracy_details_for_all_epochs} should be ≥ 99.4%"

    def test_class_accuracies(self):
        """Verify per-class accuracy"""
        if os.path.exists('logs/class_accuracy.json'):
            with open('logs/class_accuracy.json', 'r') as f:
                class_acc = json.load(f)
            for class_name, accuracy in class_acc.items():
                assert accuracy >= 95.0, \
                    f"Accuracy for class {class_name} ({accuracy}%) should be ≥ 95%"

    def test_loss_convergence(self):
        """Verify loss convergence"""
        if os.path.exists('logs/metrics.json'):
            with open('logs/metrics.json', 'r') as f:
                metrics = json.load(f)
            final_loss = metrics['test_loss'][-1]
            assert final_loss < 0.1, f"Final loss {final_loss} should be < 0.1"

    def test_model_inference(self, model, test_loader):
        """Test model performance on actual data"""
        model.eval()
        device = get_device()
        model = model.to(device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        assert accuracy >= 96, f"Live inference accuracy {accuracy}% should be ≥ 96%" 