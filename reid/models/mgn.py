import copy
import torch
from torch import nn, optim
from torchvision.models.resnet import resnet50, Bottleneck

# https://github.com/levyfan/reid-mgn/blob/master/mgn/mgn.py

__all__ = ['mgn']
#parser.add_argument('--model', choices=['mgn', 'p1_single', 'p2_single', 'p3_single'], default='mgn')


class MGN(nn.Module):

    def __init__(self, cut_at_pooling, pretrained=True, num_classes=256):
        super(MGN, self).__init__()
        self.cut_at_pooling = cut_at_pooling
        self.pretrained = pretrained
        resnet = resnet50(pretrained=pretrained)
        self.num_classes = num_classes

        # backbone
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3[0],  # res_conv4_1
        )
        # Multiple Granularity Network -> Network Architecture: The difference is that there is no down-sampling
        # in res_conv5_1 block

        # res_conv4x
        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        # res_conv5 global
        res_g_conv5 = resnet.layer4
        # res_conv5 part
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        # mgn part-1 global
        self.part_1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        # mgn part-2
        self.part_2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        # mgn part-3
        self.part_3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        # global max pooling
        self.maxpool_zg_part_1 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_part_2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_part_3 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zpart_2 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zpart_3 = nn.MaxPool2d(kernel_size=(8, 8))

        # The 1 x 1 convolutions for dimension reduction and fully connected layers for identity
        # prediction in each branch DO NOT share weights with each other.

        # Different branches in the network are all initialized with the
        # same pretrained weights of the corresponding layers after the res conv4 1 block.

        # conv1 reduce
        reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        # fc softmax loss
        #self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        #self.fc_id_2048_1 = nn.Linear(2048, num_classes)
        #self.fc_id_2048_2 = nn.Linear(2048, num_classes)

        #self.fc_id_256_1_0 = nn.Linear(256, num_classes)
        #self.fc_id_256_1_1 = nn.Linear(256, num_classes)
        #self.fc_id_256_2_0 = nn.Linear(256, num_classes)
        #self.fc_id_256_2_1 = nn.Linear(256, num_classes)
        #self.fc_id_256_2_2 = nn.Linear(256, num_classes)

        #self._init_fc(self.fc_id_2048_0)
        #self._init_fc(self.fc_id_2048_1)
        #self._init_fc(self.fc_id_2048_2)

        #self._init_fc(self.fc_id_256_1_0)
        #self._init_fc(self.fc_id_256_1_1)
        #self._init_fc(self.fc_id_256_2_0)
        #self._init_fc(self.fc_id_256_2_1)
        #self._init_fc(self.fc_id_256_2_2)


    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)

        predict = []
        triplet_losses = []
        softmax_losses = []

        # Part 1
        part_1 = self.part_1(x)

        zg_part_1 = self.maxpool_zg_part_1(part_1)  # z_g^G
        fg_part_1 = self.reduction_0(zg_part_1).squeeze(dim=3).squeeze(dim=2)  # f_g^G, L_triplet^G
        #l_part_1 = self.fc_id_2048_0(zg_part_1.squeeze(dim=3).squeeze(dim=2))  # L_softmax^G
        #print(fg_part_1.size())
        predict.append(fg_part_1)
        #triplet_losses.append(fg_part_1)
        #softmax_losses.append(l_part_1)

        # Part 2
        part_2 = self.part_2(x)

        #zg_part_2 = self.maxpool_zg_part_2(part_2)  # z_g^P2
        #fg_part_2 = self.reduction_1(zg_part_2).squeeze(dim=3).squeeze(dim=2)  # f_g^P2, L_triplet^P2
        #l_part_2 = self.fc_id_2048_1(zg_part_2.squeeze(dim=3).squeeze(dim=2))  # L_softmax^P2

        zp2 = self.maxpool_zpart_2(part_2)
        z0_p2 = zp2[:, :, 0:1, :]  # z_p0^P2
        z1_p2 = zp2[:, :, 1:2, :]  # z_p1^P2
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)  # f_p0^P2
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)  # f_p1^P2
        #l0_p2 = self.fc_id_256_1_0(f0_p2)  # L_softmax0^P2
        #l1_p2 = self.fc_id_256_1_1(f1_p2)  # L_softmax1^P2
        #print(f0_p2.size(), f1_p2.size())
        #predict.extend([fg_part_2, f0_p2, f1_p2])
        predict.extend([f0_p2, f1_p2])
        #triplet_losses.append(fg_part_2)
        #softmax_losses.extend([l_part_2, l0_p2, l1_p2])

        # Part 3
        part_3 = self.part_3(x)

        #zg_part_3 = self.maxpool_zg_part_3(part_3)  # z_g^P3
        #fg_part_3 = self.reduction_2(zg_part_3).squeeze(dim=3).squeeze(dim=2)  # f_g^P3, L_triplet^P3
        #l_part_3 = self.fc_id_2048_2(zg_part_3.squeeze(dim=3).squeeze(dim=2))  # L_softmax^P3

        zp3 = self.maxpool_zpart_3(part_3)
        z0_p3 = zp3[:, :, 0:1, :]  # z_p0^P3
        z1_p3 = zp3[:, :, 1:2, :]  # z_p1^P3
        z2_p3 = zp3[:, :, 2:3, :]  # z_p2^P3
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)  # f_p0^P3
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)  # f_p1^P3
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)  # f_p2^P3
        #l0_p3 = self.fc_id_256_2_0(f0_p3)  # L_softmax0^P3
        #l1_p3 = self.fc_id_256_2_1(f1_p3)  # L_softmax1^P3
        #l2_p3 = self.fc_id_256_2_2(f2_p3)  # L_softmax2^P3
        #predict.extend([fg_part_3, f0_p3, f1_p3, f2_p3])
        predict.extend([f0_p3, f1_p3, f2_p3])
        #triplet_losses.append(fg_part_3)
        #softmax_losses.extend([l_part_3, l0_p3, l1_p3, l2_p3])

        # Final Step
        # To obtain the most powerful discrimination, all the features reduced to 256-dim are concatenated as the final feature.
        predict = torch.cat(predict, dim=1)

        #print(x.size())
        #x = x.view(x.size(0), -1)
        #print(x.size())
        return predict # predict, triplet_losses, softmax_losses


def mgn(**kwargs):
    return MGN(**kwargs)


"""
def run():

    mgn = MGN(num_classes=len(train_dataset.unique_ids)).to(DEVICE)
    cross_entropy_loss = nn.CrossEntropyLoss()
    triplet_semihard_loss = TripletSemihardLoss(margin=1.2)
    optimizer = optim.SGD(mgn.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

    epochs = 80
    for epoch in range(epochs):
        mgn.train()

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = mgn(inputs)
            losses = [triplet_semihard_loss(output, labels) for output in outputs[1]] + \
                     [cross_entropy_loss(output, labels) for output in outputs[2]]
            loss = sum(losses) / len(losses)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            print('%d/%d - %d/%d - loss: %f' % (epoch + 1, epochs, i, len(train_loader), loss.item()))
        print('epoch: %d/%d - loss: %f' % (epoch + 1, epochs, running_loss / len(train_loader))) """