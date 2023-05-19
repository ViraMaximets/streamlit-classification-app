import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

# Model
NUM_CLASSES = 10
RESNET_PATH = 'resnet2-model-checkpoint.pt'

BATCH_SIZE = 20
STEP_SIZE = 25
LR = 1e-4
EPOCH_NUMBER = 10

DEVICE = torch.device("cpu")


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def initialize_resnet18(frz=True):
    net = resnet18(weights=ResNet18_Weights.DEFAULT)
    if frz:
        freeze(net)

    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features, NUM_CLASSES)

    return net


class ResNetModel:
    def __init__(self, new_model=False, frz=False):
        self.frz = frz

        if (new_model):
            self.net = initialize_resnet18(self.use_pretrained, self.frz)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
        else:
            self.net, self.criterion, self.optimizer = self.load_modell(
                initialize_resnet18(frz=self.frz), RESNET_PATH)

    def load_modell(self, net, path_pt=RESNET_PATH):
        checkpoint = torch.load(path_pt, map_location=DEVICE)
        net = net.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)

        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        net.eval()
        return net, criterion, optimizer

    def predict(self, img):
        return nn.Softmax()(self.net(img.unsqueeze(0).to(torch.device("cpu")))) * 100
