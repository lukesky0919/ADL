import torch.nn as nn
import torch
from  selflayer import *
from rcmdataprocess import dataprocess ,datareset
from aldlayer.aldv1_conti import *
from aldlayer.aldv9_conti import *
from aldlayer.aldv8_conti import *
from aldlayer.ald_pic import *
from utils import  *
from fusion_feature_extractor import *
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

class model4eachpic(nn.Module):
    def __init__(self,num_train,dset,back_bone):
        super(model4eachpic, self).__init__()
        # model_path = 'bestacc.pth'
        # load_model(model_name='resnet50_pmg', pretrain=True, require_grad=True)
        self.model = load_model_ald(model_name=back_bone,dset=dset)
        self.layer1 = aldlayer(8, num_train)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)


    def forward(self,x,size,batch_seq,batchsize,train):
        # result = 0
        if train:
            if size == '8' :

                x ,result = self.layer1(x,batch_seq,batchsize)

            elif size == '4' :
                # x = x
                x, result = self.layer2(x, batch_seq, batchsize)

            elif size == '2':


                x, result = self.layer3(x, batch_seq, batchsize)

        xc1, xc2, xc3, x_concat =  self.model(x)

        if size== 'original' :
            return xc1, xc2, xc3, x_concat
        else:
            return xc1, xc2, xc3, x_concat, result

class model_eachpic_tranfea(nn.Module):
    def __init__(self,num_train,dset,back_bone):
        super(model_eachpic_tranfea, self).__init__()
        # model_path = 'bestacc.pth'
        # load_model(model_name='resnet50_pmg', pretrain=True, require_grad=True)
        self.model = load_model_ald(model_name=back_bone,dset=dset)
        self.layer1 = aldlayer(8, num_train)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)


    def forward(self,x,size,batch_seq,batchsize,train,x_trans_feature):
        # result = 0
        if train:
            if size == '8' :

                x ,result = self.layer1(x,batch_seq,batchsize)

            elif size == '4' :
                # x = x
                x, result = self.layer2(x, batch_seq, batchsize)

            elif size == '2':


                x, result = self.layer3(x, batch_seq, batchsize)

        xc1, xc2, xc3, x_concat =  self.model(x,x_trans_feature)

        if size== 'original' :
            return xc1, xc2, xc3, x_concat
        else:
            return xc1, xc2, xc3, x_concat, result

class only8_8model(nn.Module):
    def __init__(self,num_classes,num_train):
        super(only8_8model, self).__init__()
        self.num_classes = num_classes
        # model_path = 'bestacc.pth'
        # load_model(model_name='resnet50_pmg', pretrain=True, require_grad=True)
        self.model = load_model_dcl(model_name='pmg_dcl', pretrain=True, require_grad=True)
        self.layer1 = aldlayer(8, num_train)

    def forward(self, x, size, batch_seq, batchsize, train):
        if train:
            if size == 8 :
                x_rcm = x
                x_rcm = dataprocess(x_rcm, [8, 8])
                x_rcm ,result = self.layer1(x_rcm,batch_seq,batchsize)
                x_rcm = datareset(x_rcm, 448, 448, [8, 8])
                x = torch.cat((x, x_rcm), dim=0)

        xc1, xc2, xc3, x_concat , xc1_swap,xc2_swap,xc3_swap, mask1, mask2,mask3 = self.model(x)

        if size== 8 :
            return xc1, xc2, xc3, x_concat ,xc1_swap,xc2_swap,xc3_swap, mask1, mask2,mask3,result
        else:
            return xc1, xc2, xc3, x_concat, xc1_swap,xc2_swap,xc3_swap, mask1, mask2,mask3

class only4_8model(nn.Module):
    def __init__(self,num_classes,num_train):
        super(only4_8model, self).__init__()
        self.num_classes = num_classes
        # model_path = 'bestacc.pth'
        # load_model(model_name='resnet50_pmg', pretrain=True, require_grad=True)
        self.model = load_model_dcl(model_name='pmg_dcl', pretrain=True, require_grad=True)
        self.layer1 = aldlayer(8, num_train)
        self.layer2 = aldlayer(4, num_train)

    def forward(self, x, size, batch_seq, batchsize, train):
        if train:
            if size == 8 :
                x_rcm = x
                x_rcm = dataprocess(x_rcm, [8, 8])
                x_rcm ,result = self.layer1(x_rcm,batch_seq,batchsize)
                x_rcm = datareset(x_rcm, 448, 448, [8, 8])
                x = torch.cat((x, x_rcm), dim=0)
            elif size == 4 :
                x_rcm = x
                x_rcm = dataprocess(x_rcm, [4, 4])
                x_rcm, result = self.layer2(x_rcm, batch_seq, batchsize)
                x_rcm = datareset(x_rcm, 448, 448, [4, 4])
                x = torch.cat((x, x_rcm), dim=0)


        xc1, xc2, xc3, x_concat , xc1_swap,xc2_swap,xc3_swap, mask1, mask2,mask3 = self.model(x)

        if (size== 8 or size == 4) :
            # print(size)
            return xc1, xc2, xc3, x_concat ,xc1_swap,xc2_swap,xc3_swap, mask1, mask2,mask3,result
        else:
            return xc1, xc2, xc3, x_concat, xc1_swap,xc2_swap,xc3_swap, mask1, mask2,mask3

class only2_8model(nn.Module):
    def __init__(self,num_classes,num_train):
        super(only2_8model, self).__init__()
        self.num_classes = num_classes
        # model_path = 'bestacc.pth'
        # load_model(model_name='resnet50_pmg', pretrain=True, require_grad=True)
        self.model = load_model_dcl(model_name='pmg_dcl', pretrain=True, require_grad=True)
        self.layer1 = aldlayer(8, num_train)
        self.layer2 = aldlayer(2, num_train)

    def forward(self, x, size, batch_seq, batchsize, train):
        if train:
            if size == 8 :
                x_rcm = x
                x_rcm = dataprocess(x_rcm, [8, 8])
                x_rcm ,result = self.layer1(x_rcm,batch_seq,batchsize)
                x_rcm = datareset(x_rcm, 448, 448, [8, 8])
                x = torch.cat((x, x_rcm), dim=0)
            elif size == 2 :
                x_rcm = x
                x_rcm = dataprocess(x_rcm, [2, 2])
                x_rcm, result = self.layer2(x_rcm, batch_seq, batchsize)
                x_rcm = datareset(x_rcm, 448, 448, [2, 2])
                x = torch.cat((x, x_rcm), dim=0)


        xc1, xc2, xc3, x_concat , xc1_swap,xc2_swap,xc3_swap, mask1, mask2,mask3 = self.model(x)

        if (size== 8 or size == 2) :
            # print(size)
            return xc1, xc2, xc3, x_concat ,xc1_swap,xc2_swap,xc3_swap, mask1, mask2,mask3,result
        else:
            return xc1, xc2, xc3, x_concat, xc1_swap,xc2_swap,xc3_swap, mask1, mask2,mask3


class model4class(nn.Module):
    def __init__(self,num_classes,dset):
        super(model4class, self).__init__()
        self.num_classes = num_classes
        self.model = load_model_ald(model_name='resnet50_pmg', dset=dset)
        # self.dropout = nn.Dropout(p=0.5)
        self.layer1 = aldlayer4class(8, num_classes)
        self.layer2 = aldlayer4class(4, num_classes)
        self.layer3 = aldlayer4class(2, num_classes)

    def forward(self, x, size,label, train):
        if train:
            if size == '8' :

                x ,result = self.layer1(x,label)
                # x = self.dropout(x)

            elif size == '4' :

                # x = self.dropout(x)
                x, result = self.layer2(x, label)

            elif size == '2':


                x, result = self.layer3(x,label)


            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size== 'original' :
            return xc1, xc2, xc3, x_concat
        else:
            return xc1, xc2, xc3, x_concat, result

class model4dataset(nn.Module):
    def __init__(self,num_dataset,dset):
        super(model4dataset, self).__init__()
        self.num_dataset = num_dataset
        self.model = load_model_ald(model_name='resnet50_pmg',dset=dset)
        # self.dropout = nn.Dropout(p=0.5)
        self.layer1 = aldlayer4dataset(8, num_dataset)
        self.layer2 = aldlayer4dataset(4, num_dataset)
        self.layer3 = aldlayer4dataset(2, num_dataset)

    def forward(self, x, size, train):
        if train:
            if size == 8 :

                x ,result = self.layer1(x)
                # x = self.dropout(x)

            elif size == 4 :

                # x = self.dropout(x)
                x, result = self.layer2(x)

            elif size == 2:


                x, result = self.layer3(x)


            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size== 1 :
            return xc1, xc2, xc3, x_concat
        else:
            return xc1, xc2, xc3, x_concat, result


class model4mulscale(nn.Module):
    # for each pic
    def __init__(self, num_train):
        super(model4mulscale, self).__init__()
        self.num_train = num_train
        self.model = load_model_ald(model_name='resnet50_pmg', pretrain=True, require_grad=True)
        # self.dropout = nn.Dropout(p=0.5)
        self.layer1 = aldlayer4mulscale( num_train)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)
        self.layer4 = aldlayer(8, num_train)
    def forward(self,x,size,batch_seq,batchsize,train):
        if train:
            if size == 'mul8' :
                x_4 ,x_8 = datamulscale4_8(x)
                x_4 ,x_8  ,resultmul_4 ,resultmul_8 = self.layer1(x_4 ,x_8,batch_seq,batchsize)
                x = datareset4_8(x_4, x_8)
            elif size == '4' :

                x, result = self.layer2(x, batch_seq, batchsize)

            elif size == '2':


                x, result = self.layer3(x, batch_seq, batchsize)

            elif size == '8' :

                x, result = self.layer4(x, batch_seq, batchsize)


            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size == '1' :
            return xc1, xc2, xc3, x_concat
        elif size == 'mul8':
            return xc1, xc2, xc3, x_concat, resultmul_4  ,resultmul_8
        else:
            return xc1, xc2, xc3, x_concat, result

class model_mulscale4class(nn.Module):
    def __init__(self, num_class):
        super(model_mulscale4class, self).__init__()
        self.num_classes = num_class
        self.model = load_model_ald(model_name='resnet50_pmg', pretrain=True, require_grad=True)
        # self.dropout = nn.Dropout(p=0.5)
        self.layer1 = aldlayer_mulscale4class( num_class)
        self.layer2 = aldlayer4class(4, self.num_classes)
        self.layer3 = aldlayer4class(2, self.num_classes)
    def forward(self,x,size,label, train):
        if train:
            if size == 8 :
                x_4 ,x_8 = datamulscale4_8(x)
                x_4 ,x_8  ,resultmul_4 ,resultmul_8 = self.layer1(x_4 ,x_8,label)
                x = datareset4_8(x_4, x_8)
            elif size == 4 :

                x, result = self.layer2(x, label)

            elif size == 2:


                x, result = self.layer3(x, label)


            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size == 1 :
            return xc1, xc2, xc3, x_concat
        elif size ==8:
            return xc1, xc2, xc3, x_concat, resultmul_4  ,resultmul_8
        else:
            return xc1, xc2, xc3, x_concat, result

class model4mulscalev2(nn.Module):
    # for each pic
    def __init__(self, num_train):
        super(model4mulscalev2, self).__init__()
        self.num_train = num_train
        self.model = load_model_aldv2(model_name='resnet50_pmg', pretrain=True, require_grad=True)
        # self.dropout = nn.Dropout(p=0.5)
        self.layer1 = ald_mulv2( num_train)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)
        self.layer4 = aldlayer(8, num_train)
    def forward(self,x,size,batch_seq,batchsize,train):
        if train:
            if size == 'mul' :
                x_boundary ,x_centre = datamulscalev2(x)
                x_boundry ,x_central  ,resultmul_boundry ,resultmul_central = self.layer1(x_boundary ,x_centre,batch_seq,batchsize)
                x = dataresetv2(x_boundary, x_central)
            elif size == '4' :

                x, result = self.layer2(x, batch_seq, batchsize)

            elif size == '2':


                x, result = self.layer3(x, batch_seq, batchsize)

            elif size == '8' :

                x, result = self.layer4(x, batch_seq, batchsize)


            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size == '1' :
            return xc1, xc2, xc3, x_concat
        elif size == 'mul':
            return xc1, xc2, xc3, x_concat, resultmul_boundry  ,resultmul_central
        else:
            return xc1, xc2, xc3, x_concat, result


class model4mulscalev4(nn.Module):
    # for each pic
    def __init__(self, num_train):
        super(model4mulscalev4, self).__init__()
        self.num_train = num_train
        self.model = load_model_ald(model_name='resnet50_pmg', pretrain=True, require_grad=True)
        # self.dropout = nn.Dropout(p=0.5)
        self.layer1 = ald_mulv4( num_train)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)
        # self.layer4 = rcmlayer(8, num_train)
    def forward(self,x,size,batch_seq,batchsize,train):
        if train:
            if size == 'mul' :
                x_boundary1 , x_boundary2,x_centre = datamulscalev4(x)
                x_boundry1,x_boundary2 ,x_central  ,resultmul_boundry1,resultmul_boundry2 ,resultmul_central = self.layer1(x_boundary1,x_boundary2,x_centre,batch_seq,batchsize)
                x = dataresetv4(x_boundary1,x_boundary2, x_central)
            elif size == '4' :

                x, result = self.layer2(x, batch_seq, batchsize)

            elif size == '2':


                x, result = self.layer3(x, batch_seq, batchsize)

            # elif size == '8' :
            #     x = dataprocess(x, [8, 8])
            #     x, result = self.layer4(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [8, 8])

            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size == '1' :
            return xc1, xc2, xc3, x_concat
        elif size == 'mul':
            return xc1, xc2, xc3, x_concat, resultmul_boundry1, resultmul_boundry2 ,resultmul_central
        else:
            return xc1, xc2, xc3, x_concat, result

class model4mulscalev1(nn.Module):
    # for each pic
    # two step
    def __init__(self, num_train,dset):
        super(model4mulscalev1, self).__init__()
        self.num_train = num_train
        self.model = load_model_ald(model_name='resnet50_pmg', dset=dset)
        # self.dropout = nn.Dropout(p=0.5)
        # 不使用 two step
        self.layer1 = aldlayer4mulscale(num_train)
        # 使用 two step
        # self.layer1 = ald_mulv1( num_train)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)
        # self.layer4 = rcmlayer(8, num_train)
    def forward(self,x,size,batch_seq,batchsize,train):
        if train:
            if size == 'mul' :
                x, result_central, result_global = self.layer1(x,batch_seq,batchsize)

            elif size == '4' :
                x, result = self.layer2(x, batch_seq, batchsize)
            elif size == '2':
                x, result = self.layer3(x, batch_seq, batchsize)

            # elif size == '8' :
            #     x = dataprocess(x, [8, 8])
            #     x, result = self.layer4(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [8, 8])

            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size == 'original' :
            return xc1, xc2, xc3, x_concat
        elif size == 'mul':
            return xc1, xc2, xc3, x_concat, result_central, result_global
        else:
            return xc1, xc2, xc3, x_concat, result





class model4mixscalev1_random(nn.Module):
    def __init__(self, num_train,num_class):
        super(model4mixscalev1_random, self).__init__()
        self.num_train = num_train
        self.num_class = num_class
        self.model = load_model(model_name='resnet50_pmg', pretrain=True, require_grad=True,num_class=num_class)
        self.layer1 = active_center()

    def forward(self, x, size, batch_seq, batchsize, train):
        if size == '8':
            x = self.layer1(x)

        xc1, xc2, xc3, x_concat = self.model(x)

        return xc1, xc2, xc3, x_concat

class model4channatt(nn.Module):
        # for each pic
        # two step
        # with channel attention
    def __init__(self, num_train,dset):
        super(model4channatt, self).__init__()
        self.num_train = num_train
        self.model = load_model_channelatt(model_name='resnet50_pmg',dset=dset)
        # self.dropout = nn.Dropout(p=0.5)
        self.layer_v0 = aldlayer(8, num_train)
        self.layer_v1 = ald_mulv1(num_train)
        self.layer_v2 = ald_mulv2(num_train)
        self.layer_v4 = ald_mulv4(num_train)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)
        # self.layer4 = rcmlayer(8, num_train)

    def forward(self, x, size, batch_seq, batchsize, train):
        if train:
            if size == 'mul':

                x_v0 = x
                x_v0, result_v0 = self.layer_v0(x_v0, batch_seq, batchsize)
                x_v1 = x
                x_v1, result_centralv1, result_globalv1 = self.layer_v1(x_v1, batch_seq, batchsize)
                x_v2 = x
                x_v2, result_centralv2, result_boundaryv2 = self.layer_v2(x_v2, batch_seq, batchsize)
                x_v4 = x
                x_v4, result_centralv4, result_boundaryv4_1,  result_boundaryv4_2 = self.layer_v4(x_v4, batch_seq, batchsize)
                x_input = [x_v0,x_v1,x_v2,x_v4]
                result = [result_v0 ,result_centralv1 ,result_globalv1, result_centralv2 ,result_boundaryv2 , result_centralv4 ,result_boundaryv4_1,  result_boundaryv4_2]
            elif size == '4':

                x_input, result = self.layer2(x, batch_seq, batchsize)

            elif size == '2':


                x_input, result = self.layer3(x, batch_seq, batchsize)
            elif size == 'original':
                x_input = x
        else:
            x_input = x

            # elif size == '8' :
            #     x = dataprocess(x, [8, 8])
            #
            #     x = datareset(x, 448, 448, [8, 8])

            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])

        xc1, xc2, xc3, x_concat = self.model(x_input)

        if size == 'original':
            return xc1, xc2, xc3, x_concat
        else:
            return xc1, xc2, xc3, x_concat, result

class Fusion_model(nn.Module):
    def __init__(self, feature_size, classes_num):
        super(Fusion_model, self).__init__()
        self.features = load_fusion_features()
        self.embed_dim = 384
        self.trans_norm = nn.LayerNorm(self.embed_dim)
        self.trans_cls_head = nn.Linear(self.embed_dim, classes_num)

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



    def forward(self, x):
        xf1, xf2, xf3, xf4, xf5 ,x_trans= self.features(x)

        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)

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
        # trans classification
        x_trans = self.trans_norm(x_trans)
        tran_cls = self.trans_cls_head(x_trans[:, 0])
        return xc1, xc2, xc3, x_concat , tran_cls



class model4mixscav8(nn.Module):
    # for each pic
    # two step
    def __init__(self, num_train,dset):
        super(model4mixscav8, self).__init__()
        self.num_train = num_train
        self.model = load_model_ald(model_name='resnet50_pmg', dset=dset)
        # self.dropout = nn.Dropout(p=0.5)
        # 使用 two step
        self.layer1 = aldlayer4decv8(num_train)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)
        # self.layer4 = rcmlayer(8, num_train)
    def forward(self,x,size,batch_seq,batchsize,train):
        if train:
            if size == 'mul' :
                # self.tensor2img(x[0], 'before' + str(batch_seq))
                x, result_central, result_global = self.layer1(x,batch_seq,batchsize)
                # self.tensor2img(x[0], 'after' + str(batch_seq))
            elif size == '4' :
                x, result = self.layer2(x, batch_seq, batchsize)
            elif size == '2':
                x, result = self.layer3(x, batch_seq, batchsize)

            # elif size == '8' :
            #     x = dataprocess(x, [8, 8])
            #     x, result = self.layer4(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [8, 8])

            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size == 'original' :
            return xc1, xc2, xc3, x_concat
        elif size == 'mul':
            return xc1, xc2, xc3, x_concat, result_central, result_global
        else:
            return xc1, xc2, xc3, x_concat, result

    def unnormalize(self,tensor, mean, std):
        # 反归一化
        # for t, m, s in zip(tensor, mean, std):
        #     t.mul_(s).add_(m)
        t = (tensor * 0.5) + 0.5

        return t

    def tensor2img(self,img, name):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unloader = transforms.ToPILImage()
        image = img.cpu().clone()  # clone the tensor
        # image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unnormalize(image, mean, std)
        image = unloader(image)
        picpath = './test19/' + name + '.jpg'
        image.save(picpath)

class model4decv1(nn.Module):
    # for each pic
    # dec on v1
    def __init__(self, num_train,dset):
        super(model4decv1, self).__init__()
        self.num_train = num_train
        self.model = load_model_ald(model_name='resnet50_pmg', dset=dset)
        # self.dropout = nn.Dropout(p=0.5)
        # 使用 two step
        self.layer1 = ald_decv1(num_train)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)
        # self.layer4 = rcmlayer(8, num_train)
    def forward(self,x,size,batch_seq,batchsize,train):
        if train:
            if size == 'mul' :
                # self.tensor2img(x[0], 'before' + str(batch_seq))
                x, result_central, result_global = self.layer1(x,batch_seq,batchsize)
                # self.tensor2img(x[0], 'after' + str(batch_seq))
            elif size == '4' :
                x, result = self.layer2(x, batch_seq, batchsize)
            elif size == '2':
                x, result = self.layer3(x, batch_seq, batchsize)

            # elif size == '8' :
            #     x = dataprocess(x, [8, 8])
            #     x, result = self.layer4(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [8, 8])

            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size == 'original' :
            return xc1, xc2, xc3, x_concat
        elif size == 'mul':
            return xc1, xc2, xc3, x_concat, result_central, result_global
        else:
            return xc1, xc2, xc3, x_concat, result

    def unnormalize(self,tensor, mean, std):
        # 反归一化
        # for t, m, s in zip(tensor, mean, std):
        #     t.mul_(s).add_(m)
        t = (tensor * 0.5) + 0.5

        return t

    def tensor2img(self,img, name):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unloader = transforms.ToPILImage()
        image = img.cpu().clone()  # clone the tensor
        # image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unnormalize(image, mean, std)
        image = unloader(image)
        picpath = './test21/' + name + '.jpg'
        image.save(picpath)

    # def feature2img(self,feature, name):
    #     mean = [0.5, 0.5, 0.5]
    #     std = [0.5, 0.5, 0.5]
    #     unloader = transforms.ToPILImage()
    #     image = img.cpu().clone()  # clone the tensor
    #     # image = image.squeeze(0)  # remove the fake batch dimension
    #     image = self.unnormalize(image, mean, std)
    #     image = unloader(image)
    #     picpath = './test21/' + name + '.jpg'
    #     image.save(picpath)

class model4fev1(nn.Module):
    # for each pic
    # feature extra on v1
    def __init__(self, num_train,dset):
        super(model4fev1, self).__init__()
        self.num_train = num_train
        self.model = load_model_ald(model_name='resnet50_pmg', dset=dset)
        # self.dropout = nn.Dropout(p=0.5)
        # 使用 two step
        self.layer1 = ald_fev1(num_train)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)
        # self.layer4 = rcmlayer(8, num_train)
    def forward(self,x,size,batch_seq,batchsize,train):
        if train:
            if size == 'mul' :
                # self.tensor2img(x[0], 'before' + str(batch_seq))
                x, result_central, result_global = self.layer1(x,batch_seq,batchsize,self.model.features)
                # self.tensor2img(x[0], 'after' + str(batch_seq))
            elif size == '4' :
                x, result = self.layer2(x, batch_seq, batchsize)
            elif size == '2':
                x, result = self.layer3(x, batch_seq, batchsize)

            # elif size == '8' :
            #     x = dataprocess(x, [8, 8])
            #     x, result = self.layer4(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [8, 8])

            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size == 'original' :
            return xc1, xc2, xc3, x_concat
        elif size == 'mul':
            return xc1, xc2, xc3, x_concat, result_central, result_global
        else:
            return xc1, xc2, xc3, x_concat, result

    def unnormalize(self,tensor, mean, std):
        # 反归一化
        # for t, m, s in zip(tensor, mean, std):
        #     t.mul_(s).add_(m)
        t = (tensor * 0.5) + 0.5

        return t

    def tensor2img(self,img, name):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unloader = transforms.ToPILImage()
        image = img.cpu().clone()  # clone the tensor
        # image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unnormalize(image, mean, std)
        image = unloader(image)
        picpath = './test24/' + name + '.jpg'
        image.save(picpath)

class model4fev1_onlycentral(nn.Module):
    # for each pic
    # feature extra on v1
    # ald learning only on central area
    def __init__(self, num_train,dset,back_bone):
        super(model4fev1_onlycentral, self).__init__()
        self.num_train = num_train
        self.model = load_model_ald(model_name=back_bone, dset=dset)
        # self.dropout = nn.Dropout(p=0.5)
        # 使用 two step
        self.layer1 = ald_fev1_onlycentre(num_train)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)
        # self.layer4 = rcmlayer(8, num_train)
    def forward(self,x,size,batch_seq,batchsize,train):
        if train:
            if size == 'mul' :
                # self.tensor2img(x[0], 'before' + str(batch_seq))
                x, result_central = self.layer1(x,batch_seq,batchsize,self.model.features)
                # self.tensor2img(x[0], 'after' + str(batch_seq))
            elif size == '4' :
                x, result = self.layer2(x, batch_seq, batchsize)
            elif size == '2':
                x, result = self.layer3(x, batch_seq, batchsize)

            # elif size == '8' :
            #     x = dataprocess(x, [8, 8])
            #     x, result = self.layer4(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [8, 8])

            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size == 'original' :
            return xc1, xc2, xc3, x_concat
        elif size == 'mul':
            return xc1, xc2, xc3, x_concat, result_central
        else:
            return xc1, xc2, xc3, x_concat, result

    def unnormalize(self,tensor, mean, std):
        # 反归一化
        # for t, m, s in zip(tensor, mean, std):
        #     t.mul_(s).add_(m)
        t = (tensor * 0.5) + 0.5

        return t

    def tensor2img(self,img, name):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unloader = transforms.ToPILImage()
        image = img.cpu().clone()  # clone the tensor
        # image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unnormalize(image, mean, std)
        image = unloader(image)
        picpath = './test25/' + name + '.jpg'
        image.save(picpath)


class model4fev8_onlycentral(nn.Module):
    # for each pic
    # feature extra on v8 for aircraft
    # ald learning only on central area
    def __init__(self, num_train,dset,back_bone):
        super(model4fev8_onlycentral, self).__init__()
        self.num_train = num_train
        self.model = load_model_ald(model_name=back_bone, dset=dset)
        # self.dropout = nn.Dropout(p=0.5)
        self.layer1 = ald_fev8_onlycentre(num_train)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)
        # self.layer4 = rcmlayer(8, num_train)
    def forward(self,x,size,batch_seq,batchsize,train):
        if train:
            if size == 'mul' :
                # self.tensor2img(x[0], 'before' + str(batch_seq))
                x, result_central = self.layer1(x,batch_seq,batchsize,self.model.features)
                # self.tensor2img(x[0], 'after' + str(batch_seq))
            elif size == '4' :
                x, result = self.layer2(x, batch_seq, batchsize)
            elif size == '2':
                x, result = self.layer3(x, batch_seq, batchsize)

            # elif size == '8' :
            #     x = dataprocess(x, [8, 8])
            #     x, result = self.layer4(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [8, 8])

            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size == 'original' :
            return xc1, xc2, xc3, x_concat
        elif size == 'mul':
            return xc1, xc2, xc3, x_concat, result_central
        else:
            return xc1, xc2, xc3, x_concat, result

    def unnormalize(self,tensor, mean, std):
        # 反归一化
        # for t, m, s in zip(tensor, mean, std):
        #     t.mul_(s).add_(m)
        t = (tensor * 0.5) + 0.5

        return t

    def tensor2img(self,img, name):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unloader = transforms.ToPILImage()
        image = img.cpu().clone()  # clone the tensor
        # image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unnormalize(image, mean, std)
        image = unloader(image)
        picpath = './test8/' + name + '.jpg'
        image.save(picpath)

class model4fev1_mixsca(nn.Module):
    # for each pic
    # feature extra on v1
    # ald learning only on central area
    # use mix scale on the pic  select 4 patch for small gra
    def __init__(self, num_train,dset):
        super(model4fev1_mixsca, self).__init__()
        self.num_train = num_train
        self.model = load_model_ald(model_name='resnet50_pmg', dset=dset)
        # self.dropout = nn.Dropout(p=0.5)
        # self.layer1 = ald_fev8_onlycentre(num_train)
        self.layer1 = ald_fev1_mixsca(num_train)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)
        # self.layer4 = rcmlayer(8, num_train)
    def forward(self,x,size,batch_seq,batchsize,train):
        if train:
            if size == 'mul' :
                self.tensor2img(x[0], 'before' + str(batch_seq))
                x, result_central ,result_global = self.layer1(x,batch_seq,batchsize,self.model.features)
                self.tensor2img(x[0], 'after' + str(batch_seq))
            elif size == '4' :
                x, result = self.layer2(x, batch_seq, batchsize)
            elif size == '2':
                x, result = self.layer3(x, batch_seq, batchsize)

            # elif size == '8' :
            #     x = dataprocess(x, [8, 8])
            #     x, result = self.layer4(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [8, 8])

            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size == 'original' :
            return xc1, xc2, xc3, x_concat
        elif size == 'mul':
            return xc1, xc2, xc3, x_concat, result_central , result_global
        else:
            return xc1, xc2, xc3, x_concat, result

    def unnormalize(self,tensor, mean, std):
        # 反归一化
        # for t, m, s in zip(tensor, mean, std):
        #     t.mul_(s).add_(m)
        t = (tensor * 0.5) + 0.5

        return t

    def tensor2img(self,img, name):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unloader = transforms.ToPILImage()
        image = img.cpu().clone()  # clone the tensor
        # image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unnormalize(image, mean, std)
        image = unloader(image)
        picpath = './testfemix/' + name + '.jpg'
        image.save(picpath)

class model4fev9_onlycentral(nn.Module):
    # for each pic
    # feature extra on v9
    # select 6 patch as our central area
    # ald learning only on central area
    def __init__(self, num_train,dset,back_bone):
        super(model4fev9_onlycentral, self).__init__()
        self.num_train = num_train
        self.model = load_model_ald(model_name=back_bone, dset=dset)
        # self.dropout = nn.Dropout(p=0.5)
        # 使用 two step
        self.layer1 = ald_fev9_onlycentre(num_train)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)
        # self.layer4 = rcmlayer(8, num_train)
    def forward(self,x,size,batch_seq,batchsize,train):
        if train:
            if size == 'mul' :
                # self.tensor2img(x[0], 'before' + str(batch_seq))
                x, result_central = self.layer1(x,batch_seq,batchsize,self.model.features)
                # self.tensor2img(x[0], 'after' + str(batch_seq))
            elif size == '4' :
                x, result = self.layer2(x, batch_seq, batchsize)
            elif size == '2':
                x, result = self.layer3(x, batch_seq, batchsize)

            # elif size == '8' :
            #     x = dataprocess(x, [8, 8])
            #     x, result = self.layer4(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [8, 8])

            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size == 'original' :
            return xc1, xc2, xc3, x_concat
        elif size == 'mul':
            return xc1, xc2, xc3, x_concat, result_central
        else:
            return xc1, xc2, xc3, x_concat, result

    def unnormalize(self,tensor, mean, std):
        # 反归一化
        # for t, m, s in zip(tensor, mean, std):
        #     t.mul_(s).add_(m)
        t = (tensor * 0.5) + 0.5

        return t

    def tensor2img(self,img, name):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unloader = transforms.ToPILImage()
        image = img.cpu().clone()  # clone the tensor
        # image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unnormalize(image, mean, std)
        image = unloader(image)
        picpath = './test25/' + name + '.jpg'
        image.save(picpath)


class model4fev1_onlycentral_conti(nn.Module):
    # for each pic
    # feature extra on v1
    # ald learning only on central area
    # central area is continuous
    # central area is 2*2
    def __init__(self, num_train,dset,back_bone):
        super(model4fev1_onlycentral_conti, self).__init__()
        self.num_train = num_train
        self.model = load_model_ald(model_name=back_bone, dset=dset)
        # self.dropout = nn.Dropout(p=0.5)
        # 使用 two step
        self.layer1 = ald_fev1_onlycentre_conti(num_train)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)
        # self.layer4 = rcmlayer(8, num_train)
    def forward(self,x,size,batch_seq,batchsize,train):
        if train:
            if size == 'mul' :
                # self.tensor2img(x[0], 'before' + str(batch_seq))
                x, result_central = self.layer1(x,batch_seq,batchsize,self.model.features)
                # self.tensor2img(x[0], 'after' + str(batch_seq))
            elif size == '4' :
                x, result = self.layer2(x, batch_seq, batchsize)
            elif size == '2':
                x, result = self.layer3(x, batch_seq, batchsize)

            # elif size == '8' :
            #     x = dataprocess(x, [8, 8])
            #     x, result = self.layer4(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [8, 8])

            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size == 'original' :
            return xc1, xc2, xc3, x_concat
        elif size == 'mul':
            return xc1, xc2, xc3, x_concat, result_central
        else:
            return xc1, xc2, xc3, x_concat, result

    def unnormalize(self,tensor, mean, std):
        # 反归一化
        # for t, m, s in zip(tensor, mean, std):
        #     t.mul_(s).add_(m)
        t = (tensor * 0.5) + 0.5

        return t

    def tensor2img(self,img, name):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unloader = transforms.ToPILImage()
        image = img.cpu().clone()  # clone the tensor
        # image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unnormalize(image, mean, std)
        image = unloader(image)
        picpath = './testconti25/' + name + '.jpg'
        image.save(picpath)

class model4fev9_onlycentral_conti(nn.Module):
    # for each pic
    # feature extra on v9
    # ald learning only on central area
    # central area is continuous
    # central area is 2*3
    def __init__(self, num_train,dset,back_bone,central_area_size):
        super(model4fev9_onlycentral_conti, self).__init__()
        self.num_train = num_train
        self.model = load_model_ald(model_name=back_bone, dset=dset)
        # self.dropout = nn.Dropout(p=0.5)
        self.layer1 = ald_fev9_onlycentre_conti(num_train,central_area_size)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)
        # self.layer4 = rcmlayer(8, num_train)
    def forward(self,x,size,batch_seq,batchsize,train):
        if train:
            if size == 'mul' :
                # self.tensor2img(x[0], 'before' + str(batch_seq))
                x, result_central = self.layer1(x,batch_seq,batchsize,self.model.features)
                # self.tensor2img(x[0], 'after' + str(batch_seq))
            elif size == '4' :
                x, result = self.layer2(x, batch_seq, batchsize)
            elif size == '2':
                x, result = self.layer3(x, batch_seq, batchsize)

            # elif size == '8' :
            #     x = dataprocess(x, [8, 8])
            #     x, result = self.layer4(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [8, 8])

            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size == 'original' :
            return xc1, xc2, xc3, x_concat
        elif size == 'mul':
            return xc1, xc2, xc3, x_concat, result_central
        else:
            return xc1, xc2, xc3, x_concat, result

    def unnormalize(self,tensor, mean, std):
        # 反归一化
        # for t, m, s in zip(tensor, mean, std):
        #     t.mul_(s).add_(m)
        t = (tensor * 0.5) + 0.5

        return t

    def tensor2img(self,img, name):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unloader = transforms.ToPILImage()
        image = img.cpu().clone()  # clone the tensor
        # image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unnormalize(image, mean, std)
        image = unloader(image)
        picpath = './testconti27/' + name + '.jpg'
        image.save(picpath)

class model4fev8_onlycentral_conti(nn.Module):
    # for each pic
    # feature extra on v9
    # ald learning only on central area
    # central area is continuous
    # central area is 2*3
    def __init__(self, num_train,dset,back_bone,central_area_size):
        super(model4fev8_onlycentral_conti, self).__init__()
        self.num_train = num_train
        self.model = load_model_ald(model_name=back_bone, dset=dset)
        # self.dropout = nn.Dropout(p=0.5)
        # 使用 two step
        self.layer1 = ald_fev8_onlycentre_conti(num_train,central_area_size)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)
        # self.layer4 = rcmlayer(8, num_train)
    def forward(self,x,size,batch_seq,batchsize,train):
        if train:
            if size == 'mul' :
                # self.tensor2img(x[0], 'before' + str(batch_seq))
                x, result_central = self.layer1(x,batch_seq,batchsize,self.model.features)
                # self.tensor2img(x[0], 'after' + str(batch_seq))
            elif size == '4' :
                x, result = self.layer2(x, batch_seq, batchsize)
            elif size == '2':
                x, result = self.layer3(x, batch_seq, batchsize)

            # elif size == '8' :
            #     x = dataprocess(x, [8, 8])
            #     x, result = self.layer4(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [8, 8])

            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size == 'original' :
            return xc1, xc2, xc3, x_concat
        elif size == 'mul':
            return xc1, xc2, xc3, x_concat, result_central
        else:
            return xc1, xc2, xc3, x_concat, result

    def unnormalize(self,tensor, mean, std):
        # 反归一化
        # for t, m, s in zip(tensor, mean, std):
        #     t.mul_(s).add_(m)
        t = (tensor * 0.5) + 0.5

        return t

    def tensor2img(self,img, name):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unloader = transforms.ToPILImage()
        image = img.cpu().clone()  # clone the tensor
        # image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unnormalize(image, mean, std)
        image = unloader(image)
        picpath = './testconti27/' + name + '.jpg'
        image.save(picpath)

class model4fev1_mulsca(nn.Module):
    # for each pic
    # feature extra on v1
    # ald learning  on central area
    # also  ald learning  on  not central area
    # 在中心区域和非中心区域同时进行打乱学习
    def __init__(self, num_train,dset,back_bone):
        super(model4fev1_mulsca, self).__init__()
        self.num_train = num_train
        self.model = load_model_ald(model_name=back_bone, dset=dset)
        # self.dropout = nn.Dropout(p=0.5)
        # 在中心区域和非中心区域同时进行打乱学习
        self.layer1 = ald_fev1_mulsca(num_train)
        self.layer2 = aldlayer(4, num_train)
        self.layer3 = aldlayer(2, num_train)
        # self.layer4 = rcmlayer(8, num_train)
    def forward(self,x,size,batch_seq,batchsize,train):
        if train:
            if size == 'mul' :
                # self.tensor2img(x[0], 'before' + str(batch_seq))
                x, result_central ,_ = self.layer1(x,batch_seq,batchsize,self.model.features)
                # self.tensor2img(x[0], 'after' + str(batch_seq))
            elif size == '4' :
                x, result = self.layer2(x, batch_seq, batchsize)
            elif size == '2':
                x, result = self.layer3(x, batch_seq, batchsize)

            # elif size == '8' :
            #     x = dataprocess(x, [8, 8])
            #     x, result = self.layer4(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [8, 8])

            # elif size == 1:
            #
            #     x = dataprocess(x, [2, 2])
            #     # x, result = self.layer3(x, batch_seq, batchsize)
            #     x = datareset(x, 448, 448, [2, 2])



        xc1, xc2, xc3, x_concat =  self.model(x)

        if size == 'original' :
            return xc1, xc2, xc3, x_concat
        elif size == 'mul':
            return xc1, xc2, xc3, x_concat, result_central
        else:
            return xc1, xc2, xc3, x_concat, result

    def unnormalize(self,tensor, mean, std):
        # 反归一化
        # for t, m, s in zip(tensor, mean, std):
        #     t.mul_(s).add_(m)
        t = (tensor * 0.5) + 0.5

        return t

    def tensor2img(self,img, name):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unloader = transforms.ToPILImage()
        image = img.cpu().clone()  # clone the tensor
        # image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unnormalize(image, mean, std)
        image = unloader(image)
        picpath = './test25/' + name + '.jpg'
        image.save(picpath)

# class PMG(nn.Module):
#     def __init__(self, model, feature_size, classes_num):
#         super(PMG, self).__init__()
#
#         self.features = model
#         self.max1 = nn.MaxPool2d(kernel_size=56, stride=56)
#         self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
#         self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
#         self.num_ftrs = 2048 * 1 * 1
#         self.elu = nn.ELU(inplace=True)
#
#         self.classifier_concat = nn.Sequential(
#             nn.BatchNorm1d(1024 * 3),
#             nn.Linear(1024 * 3, feature_size),
#             nn.BatchNorm1d(feature_size),
#             nn.ELU(inplace=True),
#             nn.Linear(feature_size, classes_num),
#         )
#
#         self.conv_block1 = nn.Sequential(
#             BasicConv(self.num_ftrs // 4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
#             BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
#         )
#         self.classifier1 = nn.Sequential(
#             nn.BatchNorm1d(self.num_ftrs // 2),
#             nn.Linear(self.num_ftrs // 2, feature_size),
#             nn.BatchNorm1d(feature_size),
#             nn.ELU(inplace=True),
#             nn.Linear(feature_size, classes_num),
#         )
#
#         self.conv_block2 = nn.Sequential(
#             BasicConv(self.num_ftrs // 2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
#             BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
#         )
#         self.classifier2 = nn.Sequential(
#             nn.BatchNorm1d(self.num_ftrs // 2),
#             nn.Linear(self.num_ftrs // 2, feature_size),
#             nn.BatchNorm1d(feature_size),
#             nn.ELU(inplace=True),
#             nn.Linear(feature_size, classes_num),
#         )
#
#         self.conv_block3 = nn.Sequential(
#             BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
#             BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
#         )
#         self.classifier3 = nn.Sequential(
#             nn.BatchNorm1d(self.num_ftrs // 2),
#             nn.Linear(self.num_ftrs // 2, feature_size),
#             nn.BatchNorm1d(feature_size),
#             nn.ELU(inplace=True),
#             nn.Linear(feature_size, classes_num),
#         )
#
#     def forward(self, x):
#         xf1, xf2, xf3, xf4, xf5 = self.features(x)
#
#         xl1 = self.conv_block1(xf3)
#         xl2 = self.conv_block2(xf4)
#         xl3 = self.conv_block3(xf5)
#
#         xl1 = self.max1(xl1)
#         xl1 = xl1.view(xl1.size(0), -1)
#         xc1 = self.classifier1(xl1)
#
#         xl2 = self.max2(xl2)
#         xl2 = xl2.view(xl2.size(0), -1)
#         xc2 = self.classifier2(xl2)
#
#         xl3 = self.max3(xl3)
#         xl3 = xl3.view(xl3.size(0), -1)
#         xc3 = self.classifier3(xl3)
#
#         x_concat = torch.cat((xl1, xl2, xl3), -1)
#         x_concat = self.classifier_concat(x_concat)
#         return xc1, xc2, xc3, x_concat
#
#
#  测试 n的值
class model4eachpic_n(nn.Module):
    def __init__(self,num_train,dset,back_bone):
        super(model4eachpic_n, self).__init__()
        # model_path = 'bestacc.pth'
        # load_model(model_name='resnet50_pmg', pretrain=True, require_grad=True)
        self.model = load_model_ald(model_name=back_bone,dset=dset)
        self.layer1 = ald_pic_n(8, num_train)
        self.layer2 = ald_pic_n(4, num_train)
        self.layer3 = ald_pic_n(2, num_train)


    def forward(self,x,size,batch_seq,batchsize,train):
        # result = 0
        if train:
            if size == '8' :

                x ,result = self.layer1(x,batch_seq,batchsize)

            elif size == '4' :
                # x = x
                x, result = self.layer2(x, batch_seq, batchsize)

            elif size == '2':


                x, result = self.layer3(x, batch_seq, batchsize)

        xc1, xc2, xc3, x_concat =  self.model(x)

        if size== 'original' :
            return xc1, xc2, xc3, x_concat
        else:
            return xc1, xc2, xc3, x_concat, result

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
