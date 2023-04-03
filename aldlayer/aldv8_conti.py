import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
# from activefunc import activercm ,getpic
from rcmdataprocess import dataprocess, datareset
from torch.nn.parameter import Parameter
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import random
from rcmdataprocess import *
import torchvision
from torchvision.models.detection.roi_heads import fastrcnn_loss


# 不用 two-step的方式做 mix-scale v8  2*4 的 中心区域
# 同时使用特征提取的方式确定中心区域
# 中心区域的形状是连续的
# 并且只在中心区域上学习打乱
class ald_fev8_onlycentre_conti(nn.Module):

    def __init__(self, num_train,central_area_size):
        super(ald_fev8_onlycentre_conti, self).__init__()
        # self.params_boundary = Parameter(torch.Tensor(num_train,12,12), requires_grad=True)
        self.params_centre = Parameter(torch.Tensor(num_train, 32, 32), requires_grad=True)
        self.avgpool1 = nn.AdaptiveAvgPool2d(4)
        self.central_area_size = central_area_size
        # torch.nn.init.normal_(self.params_boundary)
        torch.nn.init.normal_(self.params_centre)

    def forward(self, x, batch_seq, batchsize, feature_extractor):
        x_centre, x_boundary, select_index = self.fea_extarea(x, feature_extractor, batch_seq)

        if x_centre.shape[0] == batchsize:
            batchnum = batchsize
        else:
            batchnum = x_centre.shape[0]

        start = batch_seq * batchsize
        end = batch_seq * batchsize + batchnum

        # first = True
        # for i in range(start,end):
        #     rcm = self.params_boundary[i]
        #     rcm = F.softmax(rcm, dim=0)
        #     b1, rcm1 = self.getrank(rcm)
        #     b2, rcm2 = self.getrank(rcm1)
        #     b3, rcm3 = self.getrank(rcm2)
        #     b4, rcm4 = self.getrank(rcm3)
        #     b5, rcm5 = self.getrank(rcm4)
        #     b6, rcm6 = self.getrank(rcm5)
        #     b7, rcm7 = self.getrank(rcm6)
        #     b8, rcm8 = self.getrank(rcm7)
        #     b9, rcm9 = self.getrank(rcm8)
        #     b10, rcm10 = self.getrank(rcm9)
        #     b11, rcm11 = self.getrank(rcm10)
        #     b12, rcm12 = self.getrank(rcm11)
        #
        #     result_r = (b1 + b2 + b3 + b4  + b5 + b6 + b7 + b8 + b9 + b10 +
        #               b11 + b12 )
        #     result_g = result_r
        #     result_b = result_r
        #     result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)
        #
        #     if first:
        #         result_boundary = torch.unsqueeze(result_single_pic, 0)
        #         first = False
        #     else:
        #         result_single_pic = torch.unsqueeze(result_single_pic,0)
        #         result_boundary = torch.cat((result_single_pic,result_boundary),0)

        first = True
        for i in range(start, end):
            rcm = self.params_centre[i]
            rcm = F.softmax(rcm, dim=0)
            b1, rcm1 = self.getrank(rcm)
            b2, rcm2 = self.getrank(rcm1)
            b3, rcm3 = self.getrank(rcm2)
            b4, rcm4 = self.getrank(rcm3)
            b5, rcm5 = self.getrank(rcm4)
            b6, rcm6 = self.getrank(rcm5)
            b7, rcm7 = self.getrank(rcm6)
            b8, rcm8 = self.getrank(rcm7)
            b9, rcm9 = self.getrank(rcm8)
            b10, rcm10 = self.getrank(rcm9)
            b11, rcm11 = self.getrank(rcm10)
            b12, rcm12 = self.getrank(rcm11)
            b13, rcm13 = self.getrank(rcm12)
            b14, rcm14 = self.getrank(rcm13)
            b15, rcm15 = self.getrank(rcm14)
            b16, rcm16 = self.getrank(rcm15)
            b17, rcm17 = self.getrank(rcm16)
            b18, rcm18 = self.getrank(rcm17)
            b19, rcm19 = self.getrank(rcm18)
            b20, rcm20 = self.getrank(rcm19)
            b21, rcm21 = self.getrank(rcm20)
            b22, rcm22 = self.getrank(rcm21)
            b23, rcm23 = self.getrank(rcm22)
            b24, rcm24 = self.getrank(rcm23)
            b25, rcm25 = self.getrank(rcm24)
            b26, rcm26 = self.getrank(rcm25)
            b27, rcm27 = self.getrank(rcm26)
            b28, rcm28 = self.getrank(rcm27)
            b29, rcm29 = self.getrank(rcm28)
            b30, rcm30 = self.getrank(rcm29)
            b31, rcm31 = self.getrank(rcm30)
            b32, rcm32 = self.getrank(rcm31)
            result_r = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 +
                        b11 + b12 + b13 + b14 + b15 + b16 + b17 + b18 + b19 + b20 +
                        b21 + b22 + b23 + b24 + b25 + b26 + b27 + b28 + b29 + b30 + b31 + b32)
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_centre = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic, 0)
                result_centre = torch.cat((result_single_pic, result_centre), 0)

        # result_boundary = result_boundary.to(torch.float32)
        result_centre = result_centre.to(torch.float32)

        x_mat_centre = x_centre.matmul(result_centre)
        # x_mat_boundary = x_boundary.matmul(result_boundary)

        x_ouput = self.datareset(x_mat_centre, x_boundary, select_index)
        return x_ouput, result_centre, select_index

    def getrank(self, rcm):
        e = 0.0000001
        # print(rcm.shape)
        size = rcm.shape[1]
        c = torch.full((size, size), -100000.0)
        c_cuda = c
        if torch.cuda.is_available():
            # c_cuda = c.to(self.device)
            c_cuda = c.cuda()
        disturb = torch.from_numpy(np.random.normal(0, 0.000001, (size, size)))
        disturb_cuda = disturb.cuda()
        rcm = rcm + disturb_cuda
        maxvalue = torch.max(rcm).detach()
        rcm_flatten = torch.squeeze(rcm.detach().reshape(1, -1), 0)
        max2, _ = torch.topk(rcm_flatten, 2)
        sec_maxvalue = max2[1]
        if (maxvalue == sec_maxvalue):
            print("-----------------------")
            print('rcm', rcm)
            print('max', maxvalue)
            print('sec', sec_maxvalue)
            print("-----------------------")
        b = torch.relu(rcm - (maxvalue + sec_maxvalue) / 2) / ((maxvalue - sec_maxvalue) / 2)
        b_max = torch.max(b).detach()
        b = b / b_max
        # b = torch.relu(rcm - maxvalue + e) / e
        b_value = b.detach()
        # print("-----------------------")
        # print("b_value")
        # print(b)
        # print("-----------------------")
        ## 判断得到值是否正确
        max_b_value = torch.max(b_value)
        if (max_b_value != 1.0):
            print(b)
            print("-----------------------")
            print('rcm', rcm)
            print('max', maxvalue)
            print('sec', sec_maxvalue)
            print("-----------------------")
        assert max_b_value == 1.0, "maxvalue of  b not 1 is : " + str(max_b_value)
        # if(max_b_value != 1.0):
        #     print(b)
        b_value = b_value.squeeze(0)
        b_value = b_value.to(torch.float32)
        cmul = torch.mm(b_value, c_cuda) + torch.mm(c_cuda, b_value)
        cmul = cmul.unsqueeze(0)
        rcmget = rcm + cmul
        return b, rcmget

    def minmaxscaler(self, rcm):
        max = torch.max(rcm)
        min = torch.min(rcm)

        return (rcm.data - min) / (max - min)

    def fea_extarea(self, batchdata, feature_extractor, batch_seq):
        _, _, _, _, fea_map = feature_extractor(batchdata)

        avgout = torch.mean(fea_map, dim=1, keepdim=True)
        # self.tensor2img(avgout[0], 'featuremap' + str(batch_seq))
        avgout = self.avgpool1(avgout)
        avgout = avgout.flatten(1)
        # _ ,selectindex = torch.topk(avgout,16,dim=1)
        datacentral_list = []
        databoundary_list = []
        if self.central_area_size == "2*4":
            patch_dict = {
                0: [0, 1, 2, 3, 4, 5, 6, 7],
                1: [4, 5, 6, 7, 8, 9, 10, 11],
                2: [8, 9, 10, 11, 12, 13, 14, 15]
            }
        else:
            patch_dict = {
                0: [0, 1, 4, 5, 8, 9, 12, 13],
                1: [1, 2, 5 ,6, 9, 10, 13, 14],
                2: [2, 3, 6, 7, 10, 11, 14, 15]
            }

        selectindex = []
        for i in range(batchdata.shape[0]):
            max_weight = 0
            for j in range(3):
                weight_sum = 0
                for p in patch_dict[j]:
                    weight_sum += avgout[i][p]
                if (max_weight < weight_sum):
                    max_weight = weight_sum
                    max_area = j

            centre_list = patch_dict[max_area]
            all_list = [i for i in range(16)]
            boundary_list = [i for i in all_list if i not in centre_list]
            # random.shuffle(boundary_list)
            selectindex.append(centre_list + boundary_list)
            indices_boundary = torch.tensor(boundary_list).cuda()
            indices_central = torch.tensor(centre_list).cuda()
            # indices_central = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]).cuda()
            # indices_boundary = torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]).cuda()
            batchdata_cur = batchdata[i]
            batchdata_cur = rearrange(batchdata_cur, 'c (h p_h) (w p_w)  -> c  (p_h p_w) (h w) ', p_h=112, p_w=112)
            patchselect_boundary = torch.index_select(batchdata_cur, 2, indices_boundary)
            patchselect_central = torch.index_select(batchdata_cur, 2, indices_central)
            # 对中心区域先复原在分割
            # 3  * 12544  * 8 -> 3 * 224 * 448
            patchselect_central = rearrange(patchselect_central, 'c (p_h p_w) (h w) -> c (h p_h) (w p_w)', h=2, w=4,
                                            p_h=112, p_w=112)
            # 3 * 224 * 448 -> 3  * 3136 * 36
            patchselect_central = rearrange(patchselect_central, 'c (h p_h) (w p_w)-> c (p_h p_w) (h w) ', h=4, w=8)

            datacentral_list.append(patchselect_central.unsqueeze(0))
            databoundary_list.append(patchselect_boundary.unsqueeze(0))
        # b * 3 * 12544 * 10
        batchdata_boundary = torch.cat(databoundary_list, dim=0)
        # b * 3 * 3136 * 24
        batchdata_central = torch.cat(datacentral_list, dim=0)
        return batchdata_central, batchdata_boundary, selectindex

    def datareset(self, batchdata_centre, batchdata_boundary, central_area):
        # batchdata_centre 转化成同一尺寸   batchdata8 b * 3 * 3136 * 36  -> b * 3  *( 112 * 112) * 8
        # 先复原 在分割
        # b * 3 * 3136 * 36  -> b * 3  * 36 * 56 * 56
        batchdata_centre = rearrange(batchdata_centre, 'b c (p_h p_w) l  -> b c l p_h p_w ', p_h=56, p_w=56)
        # b * 3 * 36 * 56 * 56 -> b * 3 * 224 * 448
        batchdata_centre = rearrange(batchdata_centre, 'b c (h w) p_h p_w -> b c (h p_h) (w p_w) ', h=4, w=8)
        # b * 3 * 224 * 448 -> b * 3 * (112 * 112) * 8
        batchdata_centre = rearrange(batchdata_centre, 'b c (h p_h) (w p_w) -> b c (p_h p_w) (h w)', h=2, w=4)
        batchdata = torch.cat((batchdata_centre, batchdata_boundary), dim=3)
        # index = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7]
        # batchdata = batchdata[:, :, :, index]
        batchsize = batchdata.shape[0]
        # 可以优化
        for i in range(batchsize):
            index = [-1 for i in range(16)]
            for j in range(8):
                # print(central_area[i][j])
                index[central_area[i][j]] = j
            start = 8
            for k in range(16):
                if index[k] == -1:
                    index[k] = start
                    start = start + 1
            # index = indexdict[central_area[i]]
            batchdata[i] = batchdata[i, :, :, index]

        x_out = rearrange(batchdata, 'b c (p_h p_w) (h w) -> b c (h p_h) (w p_w)', h=4, w=4, p_h=112, p_w=112)
        return x_out

    def unnormalize(self, tensor, mean, std):
        # 反归一化
        # for t, m, s in zip(tensor, mean, std):
        #     t.mul_(s).add_(m)
        t = (tensor * 0.5) + 0.5

        return t

    def tensor2img(self, img, name):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unloader = transforms.ToPILImage()
        image = img.cpu().clone()  # clone the tensor
        # image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unnormalize(image, mean, std)
        image = unloader(image)
        picpath = './test24/' + name + '.jpg'
        image.save(picpath)
