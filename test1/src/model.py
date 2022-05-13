import torch
import torchvision.models as models
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models.efficientnet import model_urls


def build_model(pretrained=True, fine_tune=True, num_classes=10, filename: str | None = None):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = models.efficientnet_b3(pretrained=pretrained)
    if isinstance(filename, str):
        model.classifier[1] = nn.Linear(in_features=1536, out_features=num_classes)  # 1536  1280
        model.load_state_dict(torch.load(filename)["model_state_dict"])
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    # Change the final classification head.
    if not isinstance(filename, str):
        model.classifier[1] = nn.Linear(in_features=1536, out_features=num_classes) # 1536  1280
    return model
