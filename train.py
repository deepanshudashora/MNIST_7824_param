import torch
import torch.optim as optim
from torchvision import datasets
from torchsummary import summary
from model import Net
from utils import data_transformation,get_device, \
                  fit_model, plot_accuracy_report, \
                  show_random_results, plot_misclassified, \
                  calculate_accuracy_per_class
import os
import json 

# CUDA?
device = get_device()
print("Available Device :: ", device)

transformation_matrix = {
                         "mean_of_data":(0.1307,),
                         "std_of_data": (0.3081,)
                         }

dataloader_kwargs = {'batch_size': 64, 'shuffle': True, 'num_workers': 1, 'pin_memory': True}


train_transforms, test_transforms = data_transformation(transformation_matrix)
train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, **dataloader_kwargs)
test_loader = torch.utils.data.DataLoader(test_data, **dataloader_kwargs)

model = Net().to(device)

training_parameters = {"learning_rate":0.017,
                       "momentum":0.9,
                       "step_size":6,
                       "gamma":0.3,
                       "max_lr":0.017,
                       "num_epochs":20
                       }
train_losses, test_losses, train_acc, test_acc = fit_model(model,training_parameters,train_loader,test_loader,device)
final_output = calculate_accuracy_per_class(model,device,test_loader,test_data)
def save_metrics(train_losses, test_losses, train_acc, test_acc):
    metrics = {
        'train_loss': train_losses,
        'test_loss': test_losses,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc
    }
    os.makedirs('logs', exist_ok=True)
    with open('logs/metrics.json', 'w') as f:
        json.dump(metrics, f)

def save_class_accuracy(class_accuracies):
    with open('logs/class_accuracy.json', 'w') as f:
        json.dump(class_accuracies, f)

# Save metrics and model
save_metrics(train_losses, test_losses, train_acc, test_acc)
torch.save(model.state_dict(), 'model.pth')
save_class_accuracy(final_output)