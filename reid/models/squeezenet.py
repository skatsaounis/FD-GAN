from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import init
import torch.utils.model_zoo as model_zoo

# https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/squeezenet.py
__all__ = ['squeezenet', 'squeezenet1_0', 'squeezenet1_1']

model_urls = {
    'squeezenet1_0':
    'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1':
    'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):
    def __init__(self, version='1_1', pretrained=True, cut_at_pooling=False, num_features=1024, num_classes=0):
        super(SqueezeNet, self).__init__()
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.num_features = num_features
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_features, kernel_size=1) #self.num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), #0.5
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        #if not self.cut_at_pooling:
            #if self.num_classes > 0:
            #    self.classifier = nn.Linear(self.num_features, self.num_classes)


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def init_pretrained_weights(model, model_url):
    # Initializes SqueezeNet model with pretrained weights.
    # Layers that don't match with pretrained layers in name/size stay unchanged.

    pretrain_dict = model_zoo.load_url(model_url, map_location=None)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def squeezenet(**kwargs):
    # default version: SqueezeNet 1_1
    return SqueezeNet(**kwargs)


def squeezenet1_1(pretrained=True, **kwargs):
    model = SqueezeNet(version='1_1', **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['squeezenet1_1'])
    return model


def squeezenet1_0(pretrained=True, **kwargs):
    model = SqueezeNet(version='1_0', **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['squeezenet1_0'])
    return model