import torch.nn as nn
import torch

from Resnet import *
# from utils import load_trans_features
from selflayer import *
import ml_collections
from transformer_feature_extractor import Transformer

class PMG(nn.Module):
    def __init__(self, model, feature_size, classes_num, num_ftrs=2048):
        super(PMG, self).__init__()

        self.features = model
        self.max1 = nn.MaxPool2d(kernel_size=56, stride=56)
        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=True)

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs // 4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs // 2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        # self.channelatt  = channelGate(4 ,16)
        # self.channelattv1 = channelGate(self.num_ftrs // 2, 16)
        # self.channelattv2 = channelGate(self.num_ftrs // 2, 16)
        # self.channelattv4 = channelGate(self.num_ftrs // 2, 16)

    def forward(self, x):
        if isinstance(x, list):
            x_list = []
            for i in range(len(x)):
                xf1, xf2, xf3, xf4, xf5 = self.features(x[i])
                x1 = self.conv_block1(xf3)

                x_list.append(x1)
            batchzize = x_list[0].shape[0]
            x_mul = torch.cat([i.view(batchzize, 1, 1792, 1792) for i in x_list], dim=1)
            # x_mul = torch.cat(x_list,dim=1)
            x_mul = self.channelatt(x_mul)
            xl1 = (x_mul[:, 0] + x_mul[:, 1] + x_mul[:, 2] + x_mul[:, 3])
            xl1 = xl1.view(batchzize, 1024, 56, 56)
            # xl1 = (x_list[0] + x_list[1] + x_list[2] + x_list[3] ) / 4
            xl1 = self.max1(xl1)
            xl1 = xl1.view(xl1.size(0), -1)
            xc1 = self.classifier1(xl1)
            xc2 = xc3 = x_concat = None
            return xc1, xc2, xc3, x_concat
        else:
            xf1, xf2, xf3, xf4, xf5 = self.features(x)
            # print(xf1.shape)
            # print(xf1.shape)
            # xf1 torch.Size([b, 64, 112, 112])
            # xf2 torch.Size([b, 256, 112, 112])
            # xf3 torch.Size([b, 512, 56, 56])
            # xf4 torch.Size([b, 1024, 28, 28])
            # xf5 torch.Size([b, 2048, 14, 14])

            xl1 = self.conv_block1(xf3)
            xl2 = self.conv_block2(xf4)
            xl3 = self.conv_block3(xf5)

            # xl1 的shape 是 b * 1024 * 56 * 56
            # xl2 的shape 是 b * 1024 * 28 * 28
            # xl3 的shape 是 b * 1024 * 14 * 14

            xl1 = self.max1(xl1)
            xl1 = xl1.view(xl1.size(0), -1)
            xc1 = self.classifier1(xl1)

            xl2 = self.max2(xl2)
            xl2 = xl2.view(xl2.size(0), -1)
            xc2 = self.classifier2(xl2)

            xl3 = self.max3(xl3)
            xl3 = xl3.view(xl3.size(0), -1)
            xc3 = self.classifier3(xl3)

            x_concat = torch.cat((xl1, xl2, xl3), -1)
            x_concat = self.classifier_concat(x_concat)
            return xc1, xc2, xc3, x_concat


class PMG_resnet18(nn.Module):
    def __init__(self, model, feature_size, classes_num, num_ftrs=2048):
        super(PMG_resnet18, self).__init__()

        self.features = model
        self.max1 = nn.MaxPool2d(kernel_size=56, stride=56)
        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = 512 * 1 * 1
        self.elu = nn.ELU(inplace=True)

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(256 * 3),
            nn.Linear(256 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs // 4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs // 2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        # self.channelatt  = channelGate(4 ,16)
        # self.channelattv1 = channelGate(self.num_ftrs // 2, 16)
        # self.channelattv2 = channelGate(self.num_ftrs // 2, 16)
        # self.channelattv4 = channelGate(self.num_ftrs // 2, 16)

    def forward(self, x):
        if isinstance(x, list):
            x_list = []
            for i in range(len(x)):
                xf1, xf2, xf3, xf4, xf5 = self.features(x[i])
                x1 = self.conv_block1(xf3)

                x_list.append(x1)
            batchzize = x_list[0].shape[0]
            x_mul = torch.cat([i.view(batchzize, 1, 1792, 1792) for i in x_list], dim=1)
            # x_mul = torch.cat(x_list,dim=1)
            x_mul = self.channelatt(x_mul)
            xl1 = (x_mul[:, 0] + x_mul[:, 1] + x_mul[:, 2] + x_mul[:, 3])
            xl1 = xl1.view(batchzize, 1024, 56, 56)
            # xl1 = (x_list[0] + x_list[1] + x_list[2] + x_list[3] ) / 4
            xl1 = self.max1(xl1)
            xl1 = xl1.view(xl1.size(0), -1)
            xc1 = self.classifier1(xl1)
            xc2 = xc3 = x_concat = None
            return xc1, xc2, xc3, x_concat
        else:
            xf1, xf2, xf3, xf4, xf5 = self.features(x)
            # print(xf1.shape)
            # print(xf1.shape)
            # xf1 torch.Size([b, 64, 112, 112])
            # xf2 torch.Size([b, 256, 112, 112])
            # xf3 torch.Size([b, 512, 56, 56])
            # xf4 torch.Size([b, 1024, 28, 28])
            # xf5 torch.Size([b, 2048, 14, 14])

            xl1 = self.conv_block1(xf3)
            xl2 = self.conv_block2(xf4)
            xl3 = self.conv_block3(xf5)

            # xl1 的shape 是 b * 1024 * 56 * 56
            # xl2 的shape 是 b * 1024 * 28 * 28
            # xl3 的shape 是 b * 1024 * 14 * 14

            xl1 = self.max1(xl1)
            xl1 = xl1.view(xl1.size(0), -1)
            xc1 = self.classifier1(xl1)

            xl2 = self.max2(xl2)
            xl2 = xl2.view(xl2.size(0), -1)
            xc2 = self.classifier2(xl2)

            xl3 = self.max3(xl3)
            xl3 = xl3.view(xl3.size(0), -1)
            xc3 = self.classifier3(xl3)

            x_concat = torch.cat((xl1, xl2, xl3), -1)
            x_concat = self.classifier_concat(x_concat)
            return xc1, xc2, xc3, x_concat


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class PMG_vgg16(nn.Module):
    def __init__(self, model, feature_size, classes_num, num_ftrs=2048):
        super(PMG_vgg16, self).__init__()

        self.features = model
        self.max1 = nn.MaxPool2d(kernel_size=56, stride=56)
        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=True)

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs // 8, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs // 4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs // 4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        # self.channelatt  = channelGate(4 ,16)
        # self.channelattv1 = channelGate(self.num_ftrs // 2, 16)
        # self.channelattv2 = channelGate(self.num_ftrs // 2, 16)
        # self.channelattv4 = channelGate(self.num_ftrs // 2, 16)

    def forward(self, x):
        if isinstance(x, list):
            x_list = []
            for i in range(len(x)):
                xf1, xf2, xf3, xf4, xf5 = self.features(x[i])
                x1 = self.conv_block1(xf3)

                x_list.append(x1)
            batchzize = x_list[0].shape[0]
            x_mul = torch.cat([i.view(batchzize, 1, 1792, 1792) for i in x_list], dim=1)
            # x_mul = torch.cat(x_list,dim=1)
            x_mul = self.channelatt(x_mul)
            xl1 = (x_mul[:, 0] + x_mul[:, 1] + x_mul[:, 2] + x_mul[:, 3])
            xl1 = xl1.view(batchzize, 1024, 56, 56)
            # xl1 = (x_list[0] + x_list[1] + x_list[2] + x_list[3] ) / 4
            xl1 = self.max1(xl1)
            xl1 = xl1.view(xl1.size(0), -1)
            xc1 = self.classifier1(xl1)
            xc2 = xc3 = x_concat = None
            return xc1, xc2, xc3, x_concat
        else:
            xf1, xf2, xf3, xf4, xf5 = self.features(x)
            # print(xf1.shape)
            # print(xf1.shape)
            # xf1 torch.Size([b, 64, 224, 224])
            # xf2 torch.Size([b, 128, 112, 112])
            # xf3 torch.Size([b, 256, 56, 56])
            # xf4 torch.Size([b, 512, 28, 28])
            # xf5 torch.Size([b, 512, 14, 14])

            xl1 = self.conv_block1(xf3)
            xl2 = self.conv_block2(xf4)
            xl3 = self.conv_block3(xf5)

            # xl1 的shape 是 b * 1024 * 56 * 56
            # xl2 的shape 是 b * 1024 * 28 * 28
            # xl3 的shape 是 b * 1024 * 14 * 14

            xl1 = self.max1(xl1)
            xl1 = xl1.view(xl1.size(0), -1)
            xc1 = self.classifier1(xl1)

            xl2 = self.max2(xl2)
            xl2 = xl2.view(xl2.size(0), -1)
            xc2 = self.classifier2(xl2)

            xl3 = self.max3(xl3)
            xl3 = xl3.view(xl3.size(0), -1)
            xc3 = self.classifier3(xl3)

            x_concat = torch.cat((xl1, xl2, xl3), -1)
            x_concat = self.classifier_concat(x_concat)
            return xc1, xc2, xc3, x_concat


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class PMG_fusion(nn.Module):
    def __init__(self, model, feature_size, classes_num, num_ftrs=2048):
        super(PMG_fusion, self).__init__()

        self.features = model
        # self.features_transformer = load_trans_features()
        # for param in self.features_transformer.parameters():
        #     param.requires_grad = False
        self.max1 = nn.MaxPool2d(kernel_size=56, stride=56)
        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=True)

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.classifier_fusion = nn.Sequential(
            nn.BatchNorm1d(1024 * 3 + 768),
            nn.Linear(1024 * 3+ 768, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs // 4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs // 2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        # self.channelatt  = channelGate(4 ,16)
        # self.channelattv1 = channelGate(self.num_ftrs // 2, 16)
        # self.channelattv2 = channelGate(self.num_ftrs // 2, 16)
        # self.channelattv4 = channelGate(self.num_ftrs // 2, 16)

    def forward(self, x,x_trans_feature):
        if isinstance(x, list):
            x_list = []
            for i in range(len(x)):
                xf1, xf2, xf3, xf4, xf5 = self.features(x[i])
                x1 = self.conv_block1(xf3)

                x_list.append(x1)
            batchzize = x_list[0].shape[0]
            x_mul = torch.cat([i.view(batchzize, 1, 1792, 1792) for i in x_list], dim=1)
            # x_mul = torch.cat(x_list,dim=1)
            x_mul = self.channelatt(x_mul)
            xl1 = (x_mul[:, 0] + x_mul[:, 1] + x_mul[:, 2] + x_mul[:, 3])
            xl1 = xl1.view(batchzize, 1024, 56, 56)
            # xl1 = (x_list[0] + x_list[1] + x_list[2] + x_list[3] ) / 4
            xl1 = self.max1(xl1)
            xl1 = xl1.view(xl1.size(0), -1)
            xc1 = self.classifier1(xl1)
            xc2 = xc3 = x_concat = None
            return xc1, xc2, xc3, x_concat
        else:

            xf1, xf2, xf3, xf4, xf5 = self.features(x)

            # print(xf1.shape)
            # print(xf1.shape)
            # xf1 torch.Size([b, 64, 112, 112])
            # xf2 torch.Size([b, 256, 112, 112])
            # xf3 torch.Size([b, 512, 56, 56])
            # xf4 torch.Size([b, 1024, 28, 28])
            # xf5 torch.Size([b, 2048, 14, 14])

            xl1 = self.conv_block1(xf3)
            xl2 = self.conv_block2(xf4)
            xl3 = self.conv_block3(xf5)

            # xl1 的shape 是 b * 1024 * 56 * 56
            # xl2 的shape 是 b * 1024 * 28 * 28
            # xl3 的shape 是 b * 1024 * 14 * 14

            xl1 = self.max1(xl1)
            xl1 = xl1.view(xl1.size(0), -1)
            xc1 = self.classifier1(xl1)

            xl2 = self.max2(xl2)
            xl2 = xl2.view(xl2.size(0), -1)
            xc2 = self.classifier2(xl2)

            xl3 = self.max3(xl3)
            xl3 = xl3.view(xl3.size(0), -1)
            xc3 = self.classifier3(xl3)
            x_fusion = 0
            if torch.is_tensor(x_trans_feature) :
                x_fusion = torch.cat((xl1, xl2, xl3, x_trans_feature), -1)
                x_fusion = self.classifier_fusion(x_fusion)
            return xc1, xc2, xc3, x_fusion


# self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
# self.classifier = nn.Linear(2048, self.num_classes, bias=False)
# #self.layer1 = self.mylayer
# self.classifier_swap = nn.Linear(2048, 2 * num_classes, bias=False)
# self.Convmask = nn.Conv2d(2048, 1, 1, stride=1, padding=0, bias=True)
# self.avgpool2 = nn.AvgPool2d(2, stride=2)
#
# x = self.model(x)
# mask = self.Convmask(x)
# mask = self.avgpool2(mask)
# mask = torch.tanh(mask)
# mask = mask.view(mask.size(0), -1)
#
# x = self.avgpool(x)
# x = x.view(x.size(0), -1)
# out = []
# # print(x.shape)
# # 用来计算普通的分类loss
# out.append(self.classifier(x))
# # 计算是否是swap过的图片的一个loss
# out.append(self.classifier_swap(x))
# # 计算 一个swap后位置之间的loss
# out.append(mask)

class PMG_dcl(nn.Module):
    def __init__(self, model, feature_size, classes_num):
        super(PMG_dcl, self).__init__()

        self.features = model

        self.max1 = nn.MaxPool2d(kernel_size=56, stride=56)
        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=True)

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs // 4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.classifier1_swap = nn.Linear(1024, 2, bias=False)

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs // 2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.classifier2_swap = nn.Linear(1024, 2, bias=False)

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )
        self.classifier3_swap = nn.Linear(1024, 2, bias=False)

        self.Convmask1 = nn.Conv2d(512, 1, 1, stride=1, padding=0, bias=True)
        self.Convmask2 = nn.Conv2d(1024, 1, 1, stride=1, padding=0, bias=True)
        self.Convmask3 = nn.Conv2d(2048, 1, 1, stride=1, padding=0, bias=True)
        self.avgpool2 = nn.AvgPool2d(7, stride=7)

    def forward(self, x):
        xf1, xf2, xf3, xf4, xf5 = self.features(x)

        mask1 = self.Convmask1(xf3)
        mask1 = self.avgpool2(mask1)
        mask1 = torch.tanh(mask1)
        mask1 = mask1.view(mask1.size(0), -1)

        mask2 = self.Convmask2(xf4)
        mask2 = self.avgpool2(mask2)
        mask2 = torch.tanh(mask2)
        mask2 = mask2.view(mask2.size(0), -1)

        mask3 = self.Convmask3(xf5)
        mask3 = self.avgpool2(mask3)
        mask3 = torch.tanh(mask3)
        mask3 = mask3.view(mask3.size(0), -1)

        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)
        # 8 * 8
        xl1 = self.max1(xl1)
        xl1 = xl1.view(xl1.size(0), -1)
        # print(xl1.shape)
        xc1 = self.classifier1(xl1)
        xc1_swap = self.classifier1_swap(xl1)
        # 4 * 4
        xl2 = self.max2(xl2)
        xl2 = xl2.view(xl2.size(0), -1)
        # print(xl2.shape)
        xc2 = self.classifier2(xl2)
        xc2_swap = self.classifier1_swap(xl2)
        # 2 * 2
        xl3 = self.max3(xl3)
        xl3 = xl3.view(xl3.size(0), -1)
        # print(xl3.shape)
        xc3 = self.classifier3(xl3)
        xc3_swap = self.classifier1_swap(xl3)

        x_concat = torch.cat((xl1, xl2, xl3), -1)

        x_concat = self.classifier_concat(x_concat)
        # print(x_concat.shape)

        return xc1, xc2, xc3, x_concat, xc1_swap, xc2_swap, xc3_swap, mask1, mask2, mask3

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.split = 'non-overlap'
    config.slide_step = 12
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    config.use_ald = False
    return config

def load_trans_features():
    # load pretrain transformer model
    config = get_b16_config()
    trans_feature_extractor = Transformer(config,448)
    checkpoint = torch.load('mul_scale_checkpoint.bin')
    pre = checkpoint['model']
    net_dict = trans_feature_extractor.state_dict()
    state_dict = {k: v for k, v in pre.items() if k in net_dict.keys()}
    net_dict.update(state_dict)
    trans_feature_extractor.load_state_dict(net_dict)
    for param in trans_feature_extractor.parameters():
        param.requires_grad = False

    return trans_feature_extractor

# net = resnet50(pretrained=False)
# net = PMG(net, 512, 200)
# x = torch.randn((2,3,448,448))
# xc1, xc2, xc3, x_concat = net(x)
