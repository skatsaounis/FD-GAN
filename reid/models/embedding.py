import math
import copy
from torch import nn
import torch
import torch.nn.functional as F


class EltwiseSubEmbed(nn.Module):
    def __init__(self, nonlinearity='square', use_batch_norm=False,
                 use_classifier=False, num_features=0, num_classes=0, use_sft=False):
        super(EltwiseSubEmbed, self).__init__()
        self.nonlinearity = nonlinearity
        self.use_sft = use_sft
        if nonlinearity is not None and nonlinearity not in ['square', 'abs']:
            raise KeyError("Unknown nonlinearity:", nonlinearity)
        self.use_batch_norm = use_batch_norm
        self.use_classifier = use_classifier
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(num_features)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if self.use_classifier:
            assert num_features > 0 and num_classes > 0
            self.classifier = nn.Linear(num_features, num_classes)
            self.classifier.weight.data.normal_(0, 0.001)
            self.classifier.bias.data.zero_()
        if self.use_sft:
            self.sft_fc1 = nn.Linear(num_features * 2, num_features // 2)
            self.sft_fc2 = nn.Linear(num_features // 2, num_classes)

    def forward(self, x1, x2):
        x = x1 - x2
        if self.nonlinearity == 'square':
            x = x.pow(2)
        elif self.nonlinearity == 'abs':
            x = x.abs()
        if self.use_batch_norm:
            x = self.bn(x)
        if self.use_classifier:
            x = x.view(x.size(0), -1)
            if self.use_sft:
                sft = spectral_feature_transform(x).cuda()
                x = torch.cat((x, sft), dim=1).cuda()
                x = self.sft_fc1(x)
                x = self.sft_fc2(x)
            else:
                x = self.classifier(x)
        else:
            x = x.sum(1)

        return x


# https://arxiv.org/pdf/1811.11405.pdf
# Code taken https://github.com/xuxu116/pytorch-reid-lite/blob/master/nets/model_main.py
# Slightly modified
def spectral_feature_transform(feature, temperature=0.2):
    # feature shape: 32 * 2048
    feature_norm = torch.norm(feature, 2, 1)
    feature_n = feature / feature_norm.view(-1, 1)

    w = torch.mm(feature_n, torch.transpose(feature_n, 0, 1)) / temperature

    w = w - torch.max(w)
    dist = F.softmax(w, 0)
    out = torch.mm(dist, feature)

    return out
