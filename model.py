import math
import torch as th
import torch.nn as nn
# import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, model_zoo, model_urls

import poincare.hype.graph as graph


# class HierarchyModel(nn.Module):
#     def __init__(self, node_size, embedding_dim):
#         super(HierarchyModel, self).__init__()
#         self.embed = nn.Embedding(node_size, embedding_dim)
#
#     def l2_norm(self, emb):
#         if len(emb.size()) > 1:
#             dim = 1
#         else:
#             dim = 0
#         return F.normalize(emb, dim=dim)
#
#     def forward(self, node):
#         embedding = self.embed(node)
#         normalized_emb = self.l2_norm(embedding)
#         return normalized_emb


class Embedding(graph.Embedding):
    def __init__(self, size, dim, manifold, sparse=True):
        super(Embedding, self).__init__(size, dim, manifold, sparse)
        self.lossfn = nn.functional.cross_entropy

    def _forward(self, e):
        return e

    def loss(self, preds, targets, weight=None, size_average=True):
        return self.lossfn(preds, targets)


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class MapFunction(nn.Module):
    def __init__(self, embedding_size):
        super(MapFunction, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Linear(512, embedding_size)
        )

    def forward(self, x):
        return self.f(x)


class Net(nn.Module):
    def __init__(self, num_classes, embedding_size):
        super(Net, self).__init__()
        self.cnn = ResNet(Bottleneck, [3, 4, 6, 3])
        self.classifier = nn.Linear(2048, num_classes)
        self.map = MapFunction(embedding_size)

    def forward(self, x):
        img_emb = self.cnn(x)
        pred = self.classifier(img_emb)
        maped_emb = self.map(img_emb)
        return pred, maped_emb

    def finetune_cnn(self, allow=True):
        for p in self.cnn.parameters():
            p.requires_grad(True)
    

def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model.add_module('fc', nn.Linear(2048, num_classes))
    if pretrained:
        model_dict = model_zoo.load_url(model_urls['resnet50'])
        model.load_state_dict(model_dict, strict=False)
    return model


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None

    if model_name == 'resnet50_cls':
        model_ft = resnet50(num_classes, use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes) # classifier

    elif model_name == 'resnet50_emb':
        model_ft = resnet50(num_classes, use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes) # image embedding

    elif model_name == 'resnet_all':
        model_ft = Net(num_classes, 3) # embedding_size = 128

    return model_ft


if __name__ == '__main__':
    net = Net(172, 128)
    for name, paras in net.named_parameters():
        print(name)
