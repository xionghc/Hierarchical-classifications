import torch as torch
import torch.nn as nn
from torchvision import models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # resnet.fc = nn.Linear(2048, 172)
        # resnet.load_state_dict(torch.load('/media/Share/hcxiong/models/model_resnet.pth'))  # load ckp.
        # print('Load pretrained success')
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear1 = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)

        features_normalized = torch.div(features, torch.norm(features, 2, 1).unsqueeze(1))
        features = self.linear1(features_normalized)
        return features


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def init_encoder_model(embed_size, pretrained):
    model_ft = EncoderCNN(embed_size)

    if pretrained:
        pre_state_dict = torch.load(pretrained)
        model_ft.load_state_dict(pre_state_dict['state_dict'], False)
    return model_ft
