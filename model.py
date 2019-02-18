import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, model_zoo, model_urls


class HierarchyModel(nn.Module):
    def __init__(self, node_size, embedding_dim):
        super(HierarchyModel, self).__init__()
        self.embed = nn.Embedding(node_size, embedding_dim)

    def l2_norm(self, emb):
        if len(emb.size()) > 1:
            dim = 1
        else:
            dim = 0
        return F.normalize(emb, dim=dim)

    def forward(self, node):
        embedding = self.embed(node)
        normalized_emb = self.l2_norm(embedding)
        return normalized_emb


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

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
    

class ResNet50_c(nn.Module):
    """ResNet-50 with classifier.

    Args:
        num_classes (int): Number of classes.
    """
    expansion = 4

    def __init__(self, pretrained= False, num_classes=1000):
        super(ResNet50_c, self).__init__()
        self.resnet = ResNet(Bottleneck, [3, 4, 6, 3])
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x
