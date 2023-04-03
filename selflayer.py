import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
# from activefunc import activercm ,getpic
from rcmdataprocess import dataprocess ,datareset
from torch.nn.parameter import Parameter
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import  random
from rcmdataprocess import *
import torchvision
from torchvision.models.detection.roi_heads import fastrcnn_loss
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=8,threshold=1000000000,linewidth=3000000)
class  aldlayer(nn.Module):

    def __init__(self,size,num_train):
        super(aldlayer, self).__init__()
        self.params = Parameter(torch.Tensor(num_train,size*size, size*size), requires_grad=True)
        self.scale = size
        self.size = size*size
        torch.nn.init.normal_(self.params)


    def forward(self, x,batch_seq,batchsize):
        # 对 self.parameter 进行 softmax
        # data pre process
        # b * 3 * 448 * 448 -> b * 3 * (448/n * 448/n ) * (n * n)
        x = rearrange(x, 'b c (h p_h) (w p_w)  -> b c (p_h p_w) (h w) ', h = self.scale, w= self.scale)
        if x.shape[0] == batchsize :
            batchnum = batchsize
        else:
            batchnum = x.shape[0]
        # self.params = F.softmax(self.params, dim=0)
        start = batch_seq*batchsize
        end   = batch_seq*batchsize + batchnum

        first = True
        for i in range(start,end):
            rcm = self.params[i]
            rcm = F.softmax(rcm, dim=0)
            if self.size == 4 :
                b1, rcm1 = self.getrank(rcm)
                b2, rcm2 = self.getrank(rcm1)
                b3, rcm3 = self.getrank(rcm2)
                b4, rcm4 = self.getrank(rcm3)
                result_r = b1 + b2 + b3 + b4
                result_g = result_r
                result_b = result_r
                result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if self.size == 16:
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
                result_r = (b1 + b2 + b3 + b4  + b5 + b6 + b7 + b8 + b9 + b10 +
                          b11 + b12 + b13 + b14 + b15 + b16)
                result_g = result_r
                result_b = result_r
                result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if self.size == 64:
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
                b33, rcm33 = self.getrank(rcm32)
                b34, rcm34 = self.getrank(rcm33)
                b35, rcm35 = self.getrank(rcm34)
                b36, rcm36 = self.getrank(rcm35)
                b37, rcm37 = self.getrank(rcm36)
                b38, rcm38 = self.getrank(rcm37)
                b39, rcm39 = self.getrank(rcm38)
                b40, rcm40 = self.getrank(rcm39)
                b41, rcm41 = self.getrank(rcm40)
                b42, rcm42 = self.getrank(rcm41)
                b43, rcm43 = self.getrank(rcm42)
                b44, rcm44 = self.getrank(rcm43)
                b45, rcm45 = self.getrank(rcm44)
                b46, rcm46 = self.getrank(rcm45)
                b47, rcm47 = self.getrank(rcm46)
                b48, rcm48 = self.getrank(rcm47)
                b49, rcm49 = self.getrank(rcm48)
                b50, rcm50 = self.getrank(rcm49)
                b51, rcm51 = self.getrank(rcm50)
                b52, rcm52 = self.getrank(rcm51)
                b53, rcm53 = self.getrank(rcm52)
                b54, rcm54 = self.getrank(rcm53)
                b55, rcm55 = self.getrank(rcm54)
                b56, rcm56 = self.getrank(rcm55)
                b57, rcm57 = self.getrank(rcm56)
                b58, rcm58 = self.getrank(rcm57)
                b59, rcm59 = self.getrank(rcm58)
                b60, rcm60 = self.getrank(rcm59)
                b61, rcm61 = self.getrank(rcm60)
                b62, rcm62 = self.getrank(rcm61)
                b63, rcm63 = self.getrank(rcm62)
                b64, rcm64 = self.getrank(rcm63)


                result_r = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 +
                          b11 + b12 + b13 + b14 + b15 + b16 + b17 + b18 + b19 + b20 +
                          b21 + b22 + b23 + b24 + b25 + b26 + b27 + b28 + b29 + b30 +
                          b31 + b32 + b33 + b34 + b35 + b36 + b37 + b38 + b39 + b40 +
                          b41 + b42 + b43 + b44 + b45 + b46 + b47 + b48 + b49 + b50 +
                          b51 + b52 + b53 + b54 + b55 + b56 + b57 + b58 + b59 + b60 +
                          b61 + b62 + b63 + b64)
                result_g = result_r
                result_b = result_r
                result_single_pic   = torch.cat((result_r,result_g,result_b),dim=0)

            if first:
                result = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic,0)
                result = torch.cat((result_single_pic,result),0)

        # print(x.shape)
        # print(result)
        result = result.to(torch.float32)
        # print(x.shape)
        # print(result.shape)
        x_mat = x.matmul(result)
        # data reset
        #  b * 3 * (448/n * 448/n ) * (n * n)  ->  b * 3 * 448 * 448
        x_mat = rearrange(x_mat, 'b c (p_h p_w) (h w)  -> b c (h  p_h) (w p_w) ', h=self.scale, w=self.scale,p_h = int(448/self.scale),p_w = int(448/self.scale) )
        return x_mat ,result
    def getrank(self,rcm):
        e = 0.0000001
        c = torch.full((self.size, self.size), -100000.0)
        c_cuda = c
        if torch.cuda.is_available() :
            # c_cuda = c.to(self.device)
            c_cuda = c.cuda()
        disturb = torch.from_numpy(np.random.normal(0, 0.000001, (self.size, self.size)))
        disturb_cuda = disturb.cuda()
        rcm = rcm + disturb_cuda
        maxvalue = torch.max(rcm).detach()
        rcm_flatten = torch.squeeze(rcm.detach().reshape(1,-1),0)
        max2, _ = torch.topk(rcm_flatten, 2)
        sec_maxvalue = max2[1]
        if (maxvalue == sec_maxvalue):
            print("-----------------------")
            print('rcm',rcm)
            print('max',maxvalue)
            print('sec',sec_maxvalue)
            print("-----------------------")
        b = torch.relu(rcm - (maxvalue + sec_maxvalue) /2 ) / ((maxvalue - sec_maxvalue) /2 )
        b_max = torch.max(b).detach()
        b = b/ b_max
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
        assert max_b_value == 1.0 , "maxvalue of  b not 1 is : " + str(max_b_value)
        # if(max_b_value != 1.0):
        #     print(b)
        b_value = b_value.squeeze(0)
        b_value = b_value.to(torch.float32)
        cmul = torch.mm(b_value, c_cuda) + torch.mm(c_cuda, b_value)
        cmul = cmul.unsqueeze(0)
        rcmget = rcm + cmul
        return b , rcmget
    def minmaxscaler(self,rcm):
        max = torch.max(rcm)
        min = torch.min(rcm)

        return (rcm.data-min) /(max-min)

class  aldlayer4class(nn.Module):

    def __init__(self,size,num_class):
        super(aldlayer4class, self).__init__()
        self.params = Parameter(torch.Tensor(num_class,size*size, size*size), requires_grad=True)
        self.size = size*size
        self.scale = size
        torch.nn.init.normal_(self.params)


    def forward(self, x,label):

        x = rearrange(x, 'b c (h p_h) (w p_w)  -> b c (p_h p_w) (h w) ', h=self.scale, w=self.scale)
        label_cpu = label.cpu()
        selectclass = label_cpu.numpy().astype(int)
        # print(selectclass)
        first = True
        for i in selectclass:
            rcm = self.params[i]
            rcm = F.softmax(rcm, dim=0)
            if self.size == 4 :
                b1, rcm1 = self.getrank(rcm)
                b2, rcm2 = self.getrank(rcm1)
                b3, rcm3 = self.getrank(rcm2)
                b4, rcm4 = self.getrank(rcm3)
                result_r = b1 + b2 + b3 + b4
                result_g = result_r
                result_b = result_r
                result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if self.size == 16:
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
                result_r = (b1 + b2 + b3 + b4  + b5 + b6 + b7 + b8 + b9 + b10 +
                          b11 + b12 + b13 + b14 + b15 + b16)
                result_g = result_r
                result_b = result_r
                result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if self.size == 64:
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
                b33, rcm33 = self.getrank(rcm32)
                b34, rcm34 = self.getrank(rcm33)
                b35, rcm35 = self.getrank(rcm34)
                b36, rcm36 = self.getrank(rcm35)
                b37, rcm37 = self.getrank(rcm36)
                b38, rcm38 = self.getrank(rcm37)
                b39, rcm39 = self.getrank(rcm38)
                b40, rcm40 = self.getrank(rcm39)
                b41, rcm41 = self.getrank(rcm40)
                b42, rcm42 = self.getrank(rcm41)
                b43, rcm43 = self.getrank(rcm42)
                b44, rcm44 = self.getrank(rcm43)
                b45, rcm45 = self.getrank(rcm44)
                b46, rcm46 = self.getrank(rcm45)
                b47, rcm47 = self.getrank(rcm46)
                b48, rcm48 = self.getrank(rcm47)
                b49, rcm49 = self.getrank(rcm48)
                b50, rcm50 = self.getrank(rcm49)
                b51, rcm51 = self.getrank(rcm50)
                b52, rcm52 = self.getrank(rcm51)
                b53, rcm53 = self.getrank(rcm52)
                b54, rcm54 = self.getrank(rcm53)
                b55, rcm55 = self.getrank(rcm54)
                b56, rcm56 = self.getrank(rcm55)
                b57, rcm57 = self.getrank(rcm56)
                b58, rcm58 = self.getrank(rcm57)
                b59, rcm59 = self.getrank(rcm58)
                b60, rcm60 = self.getrank(rcm59)
                b61, rcm61 = self.getrank(rcm60)
                b62, rcm62 = self.getrank(rcm61)
                b63, rcm63 = self.getrank(rcm62)
                b64, rcm64 = self.getrank(rcm63)


                result_r = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 +
                          b11 + b12 + b13 + b14 + b15 + b16 + b17 + b18 + b19 + b20 +
                          b21 + b22 + b23 + b24 + b25 + b26 + b27 + b28 + b29 + b30 +
                          b31 + b32 + b33 + b34 + b35 + b36 + b37 + b38 + b39 + b40 +
                          b41 + b42 + b43 + b44 + b45 + b46 + b47 + b48 + b49 + b50 +
                          b51 + b52 + b53 + b54 + b55 + b56 + b57 + b58 + b59 + b60 +
                          b61 + b62 + b63 + b64)
                result_g = result_r
                result_b = result_r
                result_single_pic   = torch.cat((result_r,result_g,result_b),dim=0)

            if first:
                result = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic,0)
                result = torch.cat((result_single_pic,result),0)

        # print(x.shape)
        # print(result)
        result = result.to(torch.float32)
        # print(x.shape)
        # print(result.shape)
        x_mat = x.matmul(result)
        x_mat = rearrange(x_mat, 'b c (p_h p_w) (h w)  -> b c (h  p_h) (w p_w) ', h=self.scale, w=self.scale,
                          p_h=int(448 / self.scale), p_w=int(448 / self.scale))
        return x_mat ,result
    def getrank(self,rcm):
        e = 0.0000001
        c = torch.full((self.size, self.size), -100000.0)
        c_cuda = c
        if torch.cuda.is_available() :
            # c_cuda = c.to(self.device)
            c_cuda = c.cuda()
        disturb = torch.from_numpy(np.random.normal(0, 0.000001, (self.size, self.size)))
        disturb_cuda = disturb.cuda()
        rcm = rcm + disturb_cuda
        maxvalue = torch.max(rcm).detach()
        rcm_flatten = torch.squeeze(rcm.detach().reshape(1,-1),0)
        max2, _ = torch.topk(rcm_flatten, 2)
        sec_maxvalue = max2[1]
        if (maxvalue == sec_maxvalue):
            print("-----------------------")
            print('rcm',rcm)
            print('max',maxvalue)
            print('sec',sec_maxvalue)
            print("-----------------------")
        b = torch.relu(rcm - (maxvalue + sec_maxvalue) /2 ) / ((maxvalue - sec_maxvalue) /2 )
        b_max = torch.max(b).detach()
        b = b/ b_max
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
        assert max_b_value == 1.0 , "maxvalue of  b not 1 is : " + str(max_b_value)
        # if(max_b_value != 1.0):
        #     print(b)
        b_value = b_value.squeeze(0)
        b_value = b_value.to(torch.float32)
        cmul = torch.mm(b_value, c_cuda) + torch.mm(c_cuda, b_value)
        cmul = cmul.unsqueeze(0)
        rcmget = rcm + cmul
        return b , rcmget
    def minmaxscaler(self,rcm):
        max = torch.max(rcm)
        min = torch.min(rcm)

        return (rcm.data-min) /(max-min)


class  aldlayer4dataset(nn.Module):

    def __init__(self,size,num_dataset):
        super(aldlayer4dataset, self).__init__()
        self.params = Parameter(torch.Tensor(num_dataset,size*size, size*size), requires_grad=True)
        self.size = size*size
        self.scale = size
        torch.nn.init.normal_(self.params)


    def forward(self, x):

        x = rearrange(x, 'b c (h p_h) (w p_w)  -> b c (p_h p_w) (h w) ', h=self.scale, w=self.scale)
        # print(selectclass)
        first = True
        rcm = self.params[0]
        rcm = F.softmax(rcm, dim=0)
        if self.size == 4 :
            b1, rcm1 = self.getrank(rcm)
            b2, rcm2 = self.getrank(rcm1)
            b3, rcm3 = self.getrank(rcm2)
            b4, rcm4 = self.getrank(rcm3)
            result_r = b1 + b2 + b3 + b4
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

        if self.size == 16:
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
            result_r = (b1 + b2 + b3 + b4  + b5 + b6 + b7 + b8 + b9 + b10 +
                      b11 + b12 + b13 + b14 + b15 + b16)
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

        if self.size == 64:
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
            b33, rcm33 = self.getrank(rcm32)
            b34, rcm34 = self.getrank(rcm33)
            b35, rcm35 = self.getrank(rcm34)
            b36, rcm36 = self.getrank(rcm35)
            b37, rcm37 = self.getrank(rcm36)
            b38, rcm38 = self.getrank(rcm37)
            b39, rcm39 = self.getrank(rcm38)
            b40, rcm40 = self.getrank(rcm39)
            b41, rcm41 = self.getrank(rcm40)
            b42, rcm42 = self.getrank(rcm41)
            b43, rcm43 = self.getrank(rcm42)
            b44, rcm44 = self.getrank(rcm43)
            b45, rcm45 = self.getrank(rcm44)
            b46, rcm46 = self.getrank(rcm45)
            b47, rcm47 = self.getrank(rcm46)
            b48, rcm48 = self.getrank(rcm47)
            b49, rcm49 = self.getrank(rcm48)
            b50, rcm50 = self.getrank(rcm49)
            b51, rcm51 = self.getrank(rcm50)
            b52, rcm52 = self.getrank(rcm51)
            b53, rcm53 = self.getrank(rcm52)
            b54, rcm54 = self.getrank(rcm53)
            b55, rcm55 = self.getrank(rcm54)
            b56, rcm56 = self.getrank(rcm55)
            b57, rcm57 = self.getrank(rcm56)
            b58, rcm58 = self.getrank(rcm57)
            b59, rcm59 = self.getrank(rcm58)
            b60, rcm60 = self.getrank(rcm59)
            b61, rcm61 = self.getrank(rcm60)
            b62, rcm62 = self.getrank(rcm61)
            b63, rcm63 = self.getrank(rcm62)
            b64, rcm64 = self.getrank(rcm63)


            result_r = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 +
                      b11 + b12 + b13 + b14 + b15 + b16 + b17 + b18 + b19 + b20 +
                      b21 + b22 + b23 + b24 + b25 + b26 + b27 + b28 + b29 + b30 +
                      b31 + b32 + b33 + b34 + b35 + b36 + b37 + b38 + b39 + b40 +
                      b41 + b42 + b43 + b44 + b45 + b46 + b47 + b48 + b49 + b50 +
                      b51 + b52 + b53 + b54 + b55 + b56 + b57 + b58 + b59 + b60 +
                      b61 + b62 + b63 + b64)
            result_g = result_r
            result_b = result_r
            result_single_pic   = torch.cat((result_r,result_g,result_b),dim=0)

        if first:
            result = torch.unsqueeze(result_single_pic, 0)
            first = False
        else:
            result_single_pic = torch.unsqueeze(result_single_pic,0)
            result = torch.cat((result_single_pic,result),0)

        # print(x.shape)
        # print(result)
        result = result.to(torch.float32)
        # print(x.shape)
        # print(result.shape)
        x_mat = x.matmul(result)
        x_mat = rearrange(x_mat, 'b c (p_h p_w) (h w)  -> b c (h  p_h) (w p_w) ', h=self.scale, w=self.scale,
                         p_h=int(448 / self.scale), p_w=int(448 / self.scale))
        return x_mat ,result
    def getrank(self,rcm):
        e = 0.0000001
        c = torch.full((self.size, self.size), -100000.0)
        c_cuda = c
        if torch.cuda.is_available() :
            # c_cuda = c.to(self.device)
            c_cuda = c.cuda()
        disturb = torch.from_numpy(np.random.normal(0, 0.000001, (self.size, self.size)))
        disturb_cuda = disturb.cuda()
        rcm = rcm + disturb_cuda
        maxvalue = torch.max(rcm).detach()
        rcm_flatten = torch.squeeze(rcm.detach().reshape(1,-1),0)
        max2, _ = torch.topk(rcm_flatten, 2)
        sec_maxvalue = max2[1]
        if (maxvalue == sec_maxvalue):
            print("-----------------------")
            print('rcm',rcm)
            print('max',maxvalue)
            print('sec',sec_maxvalue)
            print("-----------------------")
        b = torch.relu(rcm - (maxvalue + sec_maxvalue) /2 ) / ((maxvalue - sec_maxvalue) /2 )
        b_max = torch.max(b).detach()
        b = b/ b_max
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
        assert max_b_value == 1.0 , "maxvalue of  b not 1 is : " + str(max_b_value)
        # if(max_b_value != 1.0):
        #     print(b)
        b_value = b_value.squeeze(0)
        b_value = b_value.to(torch.float32)
        cmul = torch.mm(b_value, c_cuda) + torch.mm(c_cuda, b_value)
        cmul = cmul.unsqueeze(0)
        rcmget = rcm + cmul
        return b , rcmget
    def minmaxscaler(self,rcm):
        max = torch.max(rcm)
        min = torch.min(rcm)

        return (rcm.data-min) /(max-min)

# 不用 two-step的方式做 mix-scale v1
class  aldlayer4mulscale(nn.Module):

    def __init__(self,num_train):
        super(aldlayer4mulscale, self).__init__()
        self.params_4 = Parameter(torch.Tensor(num_train,12,12), requires_grad=True)
        self.params_8 = Parameter(torch.Tensor(num_train,16,16), requires_grad=True)
        torch.nn.init.normal_(self.params_4)
        torch.nn.init.normal_(self.params_4)


    def forward(self, x,batch_seq,batchsize):

        x_4 ,x_8 =datamulscale4_8(x)

        if x_4.shape[0] == batchsize :
            batchnum = batchsize
        else:
            batchnum = x_4.shape[0]

        start = batch_seq*batchsize
        end   = batch_seq*batchsize + batchnum

        first = True
        for i in range(start,end):
            rcm = self.params_4[i]

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

            result_r = (b1 + b2 + b3 + b4  + b5 + b6 + b7 + b8 + b9 + b10 +
                      b11 + b12 )
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_4 = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic,0)
                result_4 = torch.cat((result_single_pic,result_4),0)

        first = True
        for i in range(start,end):
            rcm = self.params_8[i]

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
            result_r = (b1 + b2 + b3 + b4  + b5 + b6 + b7 + b8 + b9 + b10 +
                      b11 + b12 + b13 + b14 + b15 + b16)
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_8 = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic,0)
                result_8 = torch.cat((result_single_pic,result_8),0)

        result_4 = result_4.to(torch.float32)
        result_8 = result_8.to(torch.float32)

        x_mat_4 = x_4.matmul(result_4)
        x_mat_8 = x_8.matmul(result_8)

        x_ouput = datareset4_8(x_mat_4,x_mat_8)
        return x_ouput,result_4,result_8
    def getrank(self,rcm):
        e = 0.0000001
        # print(rcm.shape)
        size = rcm.shape[1]
        c = torch.full((size, size), -100000.0)
        c_cuda = c
        if torch.cuda.is_available() :
            # c_cuda = c.to(self.device)
            c_cuda = c.cuda()
        disturb = torch.from_numpy(np.random.normal(0, 0.000001, (size, size)))
        disturb_cuda = disturb.cuda()
        rcm = rcm + disturb_cuda
        maxvalue = torch.max(rcm).detach()
        rcm_flatten = torch.squeeze(rcm.detach().reshape(1,-1),0)
        max2, _ = torch.topk(rcm_flatten, 2)
        sec_maxvalue = max2[1]
        if (maxvalue == sec_maxvalue):
            print("-----------------------")
            print('rcm',rcm)
            print('max',maxvalue)
            print('sec',sec_maxvalue)
            print("-----------------------")
        b = torch.relu(rcm - (maxvalue + sec_maxvalue) /2 ) / ((maxvalue - sec_maxvalue) /2 )
        b_max = torch.max(b).detach()
        b = b/ b_max
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
        assert max_b_value == 1.0 , "maxvalue of  b not 1 is : " + str(max_b_value)
        # if(max_b_value != 1.0):
        #     print(b)
        b_value = b_value.squeeze(0)
        b_value = b_value.to(torch.float32)
        cmul = torch.mm(b_value, c_cuda) + torch.mm(c_cuda, b_value)
        cmul = cmul.unsqueeze(0)
        rcmget = rcm + cmul
        return b , rcmget
    def minmaxscaler(self,rcm):
        max = torch.max(rcm)
        min = torch.min(rcm)

        return (rcm.data-min) /(max-min)
# x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
# print(x)
# print(x.data)
# layer = rcmlayer(49,500)
# x_rcm ,result = layer(x_rcm,batch_seq,batchsize)

# 每一个类别一种打乱方式 混合尺寸的图片
class  aldlayer_mulscale4class(nn.Module):

    def __init__(self,num_class):
        super(aldlayer_mulscale4class, self).__init__()
        self.params_4 = Parameter(torch.Tensor(num_class,12,12), requires_grad=True)
        self.params_8 = Parameter(torch.Tensor(num_class,16,16), requires_grad=True)
        torch.nn.init.normal_(self.params_4)
        torch.nn.init.normal_(self.params_8)


    def forward(self, x_4,x_8,label):

        label_cpu = label.cpu()
        selectclass = label_cpu.numpy().astype(int)

        first = True
        for i in selectclass:
            rcm = self.params_4[i]

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

            result_r = (b1 + b2 + b3 + b4  + b5 + b6 + b7 + b8 + b9 + b10 +
                      b11 + b12 )
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_4 = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic,0)
                result_4 = torch.cat((result_single_pic,result_4),0)

        first = True
        for i in selectclass:
            rcm = self.params_8[i]

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
            result_r = (b1 + b2 + b3 + b4  + b5 + b6 + b7 + b8 + b9 + b10 +
                      b11 + b12 + b13 + b14 + b15 + b16)
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_8 = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic,0)
                result_8 = torch.cat((result_single_pic,result_8),0)

        result_4 = result_4.to(torch.float32)
        result_8 = result_8.to(torch.float32)

        x_mat_4 = x_4.matmul(result_4)
        x_mat_8 = x_8.matmul(result_8)
        return x_mat_4 ,x_mat_8,result_4,result_8
    def getrank(self,rcm):
        e = 0.0000001
        # print(rcm.shape)
        size = rcm.shape[1]
        c = torch.full((size, size), -100000.0)
        c_cuda = c
        if torch.cuda.is_available() :
            # c_cuda = c.to(self.device)
            c_cuda = c.cuda()
        disturb = torch.from_numpy(np.random.normal(0, 0.000001, (size, size)))
        disturb_cuda = disturb.cuda()
        rcm = rcm + disturb_cuda
        maxvalue = torch.max(rcm).detach()
        rcm_flatten = torch.squeeze(rcm.detach().reshape(1,-1),0)
        max2, _ = torch.topk(rcm_flatten, 2)
        sec_maxvalue = max2[1]
        if (maxvalue == sec_maxvalue):
            print("-----------------------")
            print('rcm',rcm)
            print('max',maxvalue)
            print('sec',sec_maxvalue)
            print("-----------------------")
        b = torch.relu(rcm - (maxvalue + sec_maxvalue) /2 ) / ((maxvalue - sec_maxvalue) /2 )
        b_max = torch.max(b).detach()
        b = b/ b_max
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
        assert max_b_value == 1.0 , "maxvalue of  b not 1 is : " + str(max_b_value)
        # if(max_b_value != 1.0):
        #     print(b)
        b_value = b_value.squeeze(0)
        b_value = b_value.to(torch.float32)
        cmul = torch.mm(b_value, c_cuda) + torch.mm(c_cuda, b_value)
        cmul = cmul.unsqueeze(0)
        rcmget = rcm + cmul
        return b , rcmget
    def minmaxscaler(self,rcm):
        max = torch.max(rcm)
        min = torch.min(rcm)

        return (rcm.data-min) /(max-min)


class ald_mulv2(nn.Module):
    # for each  pic

    def __init__(self,num_train):
        super(ald_mulv2, self).__init__()
        self.params_boundary = Parameter(torch.Tensor(num_train,24,24), requires_grad=True)
        self.params_central = Parameter(torch.Tensor(num_train,16,16), requires_grad=True)
        torch.nn.init.normal_(self.params_boundary)
        torch.nn.init.normal_(self.params_central)

    def forward(self, x, batch_seq, batchsize):
        x_boundary, x_central = datamulscalev2(x)
        if x_boundary.shape[0] == batchsize :
            batchnum = batchsize
        else:
            batchnum = x_boundary.shape[0]

        start = batch_seq*batchsize
        end   = batch_seq*batchsize + batchnum

        first = True
        for i in range(start, end):
            rcm = self.params_boundary[i]

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
            result_r = (b1 + b2 + b3 + b4  + b5 + b6 + b7 + b8 + b9 + b10 +
                        b11 + b12 + b13 + b14 + b15 + b16 + b17 + b18 + b19 + b20 +
                          b21 + b22 + b23 + b24)
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_boundary = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic, 0)
                result_boundary = torch.cat((result_single_pic, result_boundary), 0)

        first = True
        for i in range(start, end):
            rcm = self.params_central[i]
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
            result_r = (b1 + b2 + b3 + b4  + b5 + b6 + b7 + b8 + b9 + b10 +
                        b11 + b12 + b13 + b14 + b15 + b16 )
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_central = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic, 0)
                result_central = torch.cat((result_single_pic, result_central), 0)

        result_boundary = result_boundary.to(torch.float32)
        result_central = result_central.to(torch.float32)

        x_mat_boundary = x_boundary.matmul(result_boundary)
        x_mat_central = x_central.matmul(result_central)
        x_output = dataresetv2 (x_mat_boundary, x_mat_central)
        return x_output , result_central, result_boundary,

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

class ald_mulv4(nn.Module):
    def __init__(self,num_train):
        super(ald_mulv4, self).__init__()
        self.params_boundary1 = Parameter(torch.Tensor(num_train,4,4), requires_grad=True)
        self.params_boundary2 = Parameter(torch.Tensor(num_train, 6, 6), requires_grad=True)
        self.params_central = Parameter(torch.Tensor(num_train,16,16), requires_grad=True)
        torch.nn.init.normal_(self.params_boundary1)
        torch.nn.init.normal_(self.params_boundary2)
        torch.nn.init.normal_(self.params_central)

    def forward(self,x, batch_seq, batchsize):
        x_boundary1, x_boundary2, x_central = datamulscalev4(x)
        if x_boundary1.shape[0] == batchsize :
            batchnum = batchsize
        else:
            batchnum = x_boundary1.shape[0]

        start = batch_seq*batchsize
        end   = batch_seq*batchsize + batchnum

        first = True
        for i in range(start, end):
            rcm = self.params_boundary1[i]

            b1, rcm1 = self.getrank(rcm)
            b2, rcm2 = self.getrank(rcm1)
            b3, rcm3 = self.getrank(rcm2)
            b4, rcm4 = self.getrank(rcm3)
            result_r = (b1 + b2 + b3 + b4 )
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_boundary1 = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic, 0)
                result_boundary1 = torch.cat((result_single_pic, result_boundary1), 0)

        first = True
        for i in range(start, end):
            rcm = self.params_boundary2[i]

            b1, rcm1 = self.getrank(rcm)
            b2, rcm2 = self.getrank(rcm1)
            b3, rcm3 = self.getrank(rcm2)
            b4, rcm4 = self.getrank(rcm3)
            b5, rcm5 = self.getrank(rcm4)
            b6, rcm6 = self.getrank(rcm5)
            result_r = (b1 + b2 + b3 + b4 + b5 + b6 )
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_boundary2 = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic, 0)
                result_boundary2 = torch.cat((result_single_pic, result_boundary2), 0)

        first = True
        for i in range(start, end):
            rcm = self.params_central[i]
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
            result_r = (b1 + b2 + b3 + b4  + b5 + b6 + b7 + b8 + b9 + b10 +
                        b11 + b12 + b13 + b14 + b15 + b16 )
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_central = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic, 0)
                result_central = torch.cat((result_single_pic, result_central), 0)

        result_boundary1 = result_boundary1.to(torch.float32)
        result_boundary2 = result_boundary2.to(torch.float32)
        result_central = result_central.to(torch.float32)

        x_mat_boundary1 = x_boundary1.matmul(result_boundary1)
        x_mat_boundary2 = x_boundary2.matmul(result_boundary2)
        x_mat_central = x_central.matmul(result_central)
        x_output = dataresetv4(x_mat_boundary1, x_mat_boundary2, x_mat_central)
        return x_output, result_central , result_boundary1,  result_boundary2

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


class ald_mulv1(nn.Module):
    def __init__(self,num_train):
        super(ald_mulv1, self).__init__()
        # two-step version
        self.params_centre = Parameter(torch.Tensor(num_train,16,16), requires_grad=True)
        self.params_global = Parameter(torch.Tensor(num_train,16,16), requires_grad=True)
        torch.nn.init.normal_(self.params_centre)
        torch.nn.init.normal_(self.params_global)

    def forward(self, x, batch_seq, batchsize):

        x_boundary , x_centre = self.datamulscale(x)

        if x_boundary.shape[0] == batchsize :
            batchnum = batchsize
        else:
            batchnum = x_boundary.shape[0]

        start = batch_seq*batchsize
        end   = batch_seq*batchsize + batchnum


        first = True
        for i in range(start, end):
            rcm = self.params_centre[i]
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
            result_r = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 +
                        b11 + b12 + b13 + b14 + b15 + b16 )
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_central = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic, 0)
                result_central = torch.cat((result_single_pic, result_central), 0)

        first = True
        for i in range(start, end):
            rcm = self.params_global[i]
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
            result_r = (b1 + b2 + b3 + b4  + b5 + b6 + b7 + b8 + b9 + b10 +
                        b11 + b12 + b13 + b14 + b15 + b16 )
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_global = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic, 0)
                result_global = torch.cat((result_single_pic, result_global), 0)

        # result_boundary1 = result_boundary1.to(torch.float32)
        # result_boundary2 = result_boundary2.to(torch.float32)
        result_central = result_central.to(torch.float32)
        result_global  = result_global.to(torch.float32)

        # x_mat_boundary1 = x_boundary1.matmul(result_boundary1)
        # x_mat_boundary2 = x_boundary2.matmul(result_boundary2)
        x_mat_central = x_centre.matmul(result_central)
        x_global = self.datarest(x_mat_central,x_boundary)
        x_mat_global = x_global.matmul(result_global)
        # b * 3  * 12544 * 16 ->  b * 3 * 448 * 448
        x_mat_global = rearrange(x_mat_global,'b c (p_h p_w) (h w) -> b c (h p_h) (w p_w)',h=4,w=4,p_h=112,p_w=112)
        return  x_mat_global ,result_central ,result_global

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

    def datarest(self,batchdata_centre,batchdata_boudary):
        # batchdata_centre 转化成同一尺寸   batchdata8 b * 3 * 3136 * 16  -> b * 3  *( 112 * 112) * 4
        # 先复原 在分割
        # b * 3 * 3136 * 16  -> b * 3  * 16 * 56 * 56
        batchdata_centre = rearrange(batchdata_centre, 'b c (p_h p_w) l  -> b c l p_h p_w ', p_h=56, p_w=56)
        # b * 3 * 16 * 56 * 56 -> b * 3 * 224 * 224
        batchdata_centre = rearrange(batchdata_centre, 'b c (h w) p_h p_w -> b c (h p_h) (w p_w) ', h=4, w=4)
        # b * 3 * 224 * 224 -> b * 3 * (112 * 112) * 4
        batchdata_centre = rearrange(batchdata_centre, 'b c (h p_h) (w p_w) -> b c (p_h p_w) (h w)', h=2, w=2)
        batchdata = torch.cat((batchdata_boudary,batchdata_centre),dim=3)
        index = [0,1,2,3,4,12,13,5,6,14,15,7,8,9,10,11]
        batchdata = batchdata[:,:,:,index]
        return  batchdata

    def datamulscale(self,batchdata):
        # b * c * h * w
        # b * 3 * 448 * 448 -> b * 3 * 16* 112 * 112
        batchdata = rearrange(batchdata, 'b c (h p_h) (w p_w)  -> b c (h w) p_h p_w ', p_h=112, p_w=112)
        # b * 3 * 16* 112 * 112 ->  b * 3 * 12544 * 16
        batchdata = rearrange(batchdata, 'b c l p_h p_w  ->  b c (p_h p_w) l  ')

        indices_boundary = torch.tensor([0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15])
        indices_boundary = indices_boundary.cuda()
        # 选取周围4*4粒度的图片
        # b * 3 * 12544 * 12
        patchselect_boundary = torch.index_select(batchdata, 3, indices_boundary)
        # 选取中心的图片做8*8粒度的分割
        # 先复原 在分割
        # b * 3 * 12544 * 4
        indices_centre = torch.tensor([5, 6, 9, 10])
        indices_centre = indices_centre.cuda()
        patchselect_centre = torch.index_select(batchdata, 3, indices_centre)
        # b * 3 * 12544 * 4 -> b * 3 * 4 * 112 * 112
        patchselect_centre = rearrange(patchselect_centre, 'b c (p_h p_w)  l ->  b c l p_h p_w  ', p_h=112, p_w=112)
        #  b * 3 * 4 * 112 * 112 ->  b * 3 * 224 * 224
        patchselect_centre = rearrange(patchselect_centre, 'b c  (h w) p_h p_w  ->  b c (h p_h) (w p_w)  ', h=2, w=2)
        #  b * 3 * 224 * 224 -> b * 3 * 16 * 56 *56
        patchselect_centre = rearrange(patchselect_centre, 'b c  (h p_h) (w p_w)  ->  b c (h w) p_h  p_w   ', h=4, w=4)
        # b * 3 * 16 * 56 *56 -> b * 3 * 3136 * 16
        patchselect_centre = rearrange(patchselect_centre, 'b c   l p_h p_w  ->  b c (p_h p_w) l   ')
        return patchselect_boundary, patchselect_centre

# select active center area for mix-scale

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class active_center(nn.Module):
    def __init__(self):
        super(active_center, self).__init__()
        vgg16 = models.vgg16(pretrained=False)
        path = './vgg16-397923af.pth'
        pre = torch.load(path)
        vgg16.load_state_dict(pre)
        vgg = vgg16.features
        if torch.cuda.is_available():
            vgg = vgg.cuda()
        for param in vgg.parameters():
            param.requires_grad_(False)
        self.feature = vgg

        self.attetion = SpatialAttentionModule()
        self.avgpool1 = nn.AdaptiveAvgPool2d(3)


    def forward(self, x):

        out = self.feature(x)
        att_weight = self.attetion(out)
        avg_att_weight = self.avgpool1(att_weight)
        batchsize = avg_att_weight.shape[0]
        for i in range(batchsize):
            maxvalue = torch.max(avg_att_weight[i]).detach()
            avg_att_flatten = torch.squeeze(avg_att_weight[i].detach().reshape(1, -1), 0)
            max2, _ = torch.topk(avg_att_flatten, 2)
            sec_maxvalue = max2[1]
            avg_att_weight[i] = torch.relu(avg_att_weight[i] - (maxvalue + sec_maxvalue) / 2) / ((maxvalue - sec_maxvalue) / 2)
            # print(avg_att_weight[i])
            att_max = torch.max(avg_att_weight[i]).detach()
            avg_att_weight[i] = avg_att_weight[i] / att_max

        left_mat = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        left_mat = left_mat.expand(batchsize,1,4,3)
        right_mat = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        right_mat = right_mat.expand(batchsize,1,3,4)


        if torch.cuda.is_available():
            left_mat_cuda = left_mat.cuda()
            right_mat_cuda = right_mat.cuda()
        if torch.cuda.is_available():
            avg_att_weight = left_mat_cuda @ avg_att_weight @ right_mat_cuda
        batchsize = x.shape[0]
        active_index = [[] for i in range(batchsize)]
        boundary_index = [[] for i in range(batchsize)]
        for b in range(batchsize):
            avg_att_flatten = torch.squeeze(avg_att_weight[b].detach().reshape(1, -1), 0)
            # print(avg_att_flatten.shape)
            len = avg_att_flatten.shape[0]
            for l in range(len) :
                if avg_att_flatten[l] == 1.0 :
                    active_index[b].append(l)
                else:
                    boundary_index[b].append(l)
        # print(active_index)
        # print(boundary_index)
        active_index = torch.tensor(active_index)
        boundary_index = torch.tensor(boundary_index)
        if torch.cuda.is_available():
            active_index = active_index.cuda()
            boundary_index = boundary_index.cuda()
        # print(active_index)
        # print(boundary_index)
        # print(x.shape)
        x = rearrange(x,'b c (h p_h) (w p_w) -> b c (h w) (p_h p_w) ' ,h=4,w=4)
        list_active = []
        list_boundary = []
        for b in range(batchsize) :
            list_active.append(torch.index_select(x[b], 1, active_index[b]))
            list_boundary.append(torch.index_select(x[b], 1, boundary_index[b]))

        x_active = torch.stack(list_active, dim=0)
        x_boundary = torch.stack(list_boundary, dim=0)
        # x_active b * 3 * 4 * (112*112)
        # x_boundary b * 3 * 12 * (112*112)
        # 先进一步打乱 x_active
        # b * 3 * 4 * 12544 ->b * 3 * 4 * 112 * 112
        x_active = rearrange(x_active, 'b c l (p_h p_w) -> b c l  p_h p_w', p_h=112, p_w=112)
        x_active = rearrange(x_active,'b c l (h p_h) (w p_w) -> b c (l h w) p_h p_w',p_h =56,p_w = 56)
        for i in range(batchsize):
            random_index = random.sample(range(0,16),16)
            x_active[i] = x_active[i,:,random_index]
        # b * 3 * 16 * 56 * 56 -> b * c * 224 * 224
        x_active = rearrange(x_active,"b c (h w) p_h p_w -> b c (h p_h) (w p_w)",h=4,w=4)
        # b * c * 224 * 224 -> b * c * 4 * (112 * 112)
        x_active = rearrange(x_active, "b c (h p_h) (w p_w) -> b c (h w) (p_h p_w)", h=2, w=2)

        x_global = torch.cat((x_active,x_boundary),dim=2)
        # x_global b * 3 * 16 *12544
        for i in range(batchsize):
            random_index = random.sample(range(0,16),16)
            x_global[i] = x_global[i,:,random_index]
        x_global = rearrange(x_global,'b c (h w) (p_h p_w) -> b c (h p_h) (w p_w)',h=4,w=4,p_h=112,p_w=112)

        return  x_global

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class channelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(channelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels % reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels % reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )


            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw


        attention_weight = torch.softmax(torch.sigmoid( channel_att_sum ),dim=1)
        # print(channel_att_sum)
        scale = channel_att_sum.unsqueeze(2).unsqueeze(3).expand_as(x)

        return attention_weight

class channelatt(nn.Module):
    def __init__(self, size):
        super(channelatt, self).__init__()
        self.att = nn.Parameter(torch.Tensor(1, size), requires_grad=True)
        nn.init.constant_(self.att, 0.25)

    def forward(self, x_mul):
        self.att = torch.softmax(self.att,dim=0)

        weight = self.att.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x_mul)
        xl1 = x_mul[:, 0] * weight[:,0] + x_mul[:, 1] * weight[:,1] + x_mul[:, 2] * weight[:,2] + x_mul[:, 3] * weight[:,3]
        return  xl1


# def faster_rcnn_detection(batchdata,obj_target):
#     # 加载pytorch自带的预训练Faster RCNN目标检测模型  obj_target 代表要检测出的物体类别 5 代表 aircraft
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#     model.cuda()
#     model.eval()
#
#     # 将图像输入神经网络模型中，得到输出
#     output = model(batchdata)
#     # 分出中心和边界区域
#     datacentral_list = []
#     databoundary_list = []
#     central_area = []
#     for i in range(len((output))):
#         labels = output[i]['labels'].cpu().detach().numpy()     # 预测每一个obj的标签
#         bboxes = output[i]['boxes'].cpu().detach().numpy()      # 预测每一个obj的边框
#         declabel = np.argwhere(labels == obj_target)
#         # 通过目标检测来选取中心区域
#         if declabel.size != 0:
#             index = declabel[0][0]
#             x1, y1, x2, y2 = bboxes[index]
#             y1 = int(y1)
#             y2 = int(y2)
#             #从上到下一共三个候选区域 选择一个目标区域所在位置最大的 三个区域中x的值是固定的 就是w
#             # 三个区域是 0-224 112-336 224-448
#             maxarea = float('-inf')
#             y = np.zeros((448,), dtype=np.int)
#             y[y1:y2] = 1
#             for i in range(3):
#                 up = i * 112
#                 down = (i+2) * 112
#                 area = sum(y[up:down])
#                 if (maxarea<area):
#                     maxarea = area
#                     maxindex = i
#             # print(maxarea)
#             print(maxindex)
#             if maxindex == 0:
#                 central_area.append(0)
#                 indices_central = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).cuda()
#                 indices_boundary = torch.tensor([8, 9, 10, 11, 12, 13, 14, 15]).cuda()
#             elif maxindex == 1:
#                 central_area.append(1)
#                 indices_central = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]).cuda()
#                 indices_boundary = torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]).cuda()
#             elif maxindex == 2:
#                 central_area.append(2)
#                 indices_central = torch.tensor([8, 9, 10, 11, 12, 13, 14, 15]).cuda()
#                 indices_boundary = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).cuda()
#             #   3 * 448 * 448 -> 3 * 12544 * 16
#             # print(batchdata[i].shape)
#             batchdata_cur = rearrange(batchdata[i], 'c (h p_h) (w p_w)  -> c  (p_h p_w) (h w) ', p_h=112, p_w=112,h=4,w=4)
#             patchselect_boundary = torch.index_select(batchdata_cur, 2, indices_boundary)
#             patchselect_central = torch.index_select(batchdata_cur, 2, indices_central)
#             # 对中心区域先复原在分割
#             # 3 * 12544 * 8 -> 3 * 224 * 448
#             patchselect_central = rearrange(patchselect_central, 'c (p_h p_w) (h w) -> c (h p_h) (w p_w)', h=2, w=4,  p_h=112, p_w=112)
#             # 3 * 224 * 448 -> 3  * 3136 * 32
#             patchselect_central = rearrange(patchselect_central, 'c (h p_h) (w p_w)-> c (p_h p_w) (h w) ', h=4, w=8)
#
#             datacentral_list.append(patchselect_central)
#             databoundary_list.append(patchselect_boundary)
#         else:
#             # 检测不出来的目标区域就默认用中间的部分
#             #   3 * 448 * 448 -> 3 * 12544 * 16
#             central_area.append(0)
#             batchdata_cur = rearrange(batchdata[i], 'c (h p_h) (w p_w)  -> c  (p_h p_w) (h w) ', p_h=112, p_w=112)
#             indices_central = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]).cuda()
#             indices_boundary = torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]).cuda()
#             patchselect_boundary = torch.index_select(batchdata_cur, 2, indices_boundary)
#             patchselect_central = torch.index_select(batchdata_cur, 2, indices_central)
#             # 对中心区域先复原在分割
#             # 3  * 12544  * 8 -> 3 * 224 * 448
#             patchselect_central = rearrange(patchselect_central ,'c (p_h p_w) (h w) -> c (h p_h) (w p_w)',h=2,w=4,p_h=112,p_w=112)
#             # 3 * 224 * 448 -> 3  * 3136 * 32
#             patchselect_central = rearrange(patchselect_central, 'c (h p_h) (w p_w)-> c (p_h p_w) (h w) ',h=4,w=8)
#
#             datacentral_list.append(patchselect_central)
#             databoundary_list.append(patchselect_boundary)
#     # b * 3 * 12544 * 8
#     batchdata_boundary = torch.stack(databoundary_list,dim=0)
#     # b * 3 * 3136 * 32
#     batchdata_central = torch.stack(datacentral_list,dim=0)
#     return batchdata_central , batchdata_boundary ,central_area

def unnormalize(tensor, mean, std):
    # 反归一化
    # for t, m, s in zip(tensor, mean, std):
    #     t.mul_(s).add_(m)
    t = (tensor * 0.5) + 0.5

    return t

def tensor2img(img, name):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    unloader = transforms.ToPILImage()
    image = img.cpu().clone()  # clone the tensor
    # image = image.squeeze(0)  # remove the fake batch dimension
    image = unnormalize(image, mean, std)
    image = unloader(image)
    picpath = './test19/' + name + '.jpg'
    image.save(picpath)

def faster_rcnn_detection(batchdata,obj_target,batch_seq):
    # 加载pytorch自带的预训练Faster RCNN目标检测模型  obj_target 代表要检测出的物体类别 5 代表 aircraft
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.cuda()
    model.eval()

    # 将图像输入神经网络模型中，得到输出
    output = model(batchdata)
    # 分出中心和边界区域
    datacentral_list = []
    databoundary_list = []
    central_area = []
    # 算出中心区域的index
    for i in range(len((output))):
        labels = output[i]['labels'].cpu().detach().numpy()     # 预测每一个obj的标签
        bboxes = output[i]['boxes'].cpu().detach().numpy()      # 预测每一个obj的边框
        declabel = np.argwhere(labels == obj_target)
        # 通过目标检测来选取中心区域
        if declabel.size != 0:
            index = declabel[0][0]
            x1, y1, x2, y2 = bboxes[index]
            y1 = int(y1)
            y2 = int(y2)
            #从上到下一共三个候选区域 选择一个目标区域所在位置最大的 三个区域中x的值是固定的 就是w
            # 三个区域是 0-224 112-336 224-448
            maxarea = float('-inf')
            y = np.zeros((448,), dtype=int)
            y[y1:y2] = 1
            for i in range(3):
                up = i * 112
                down = (i+2) * 112
                area = sum(y[up:down])
                if (maxarea<area):
                    maxarea = area
                    maxindex = i
            # print(maxarea)
            # print(maxindex)
            central_area.append(maxindex)
        else:
            central_area.append(1)
    for i in range(batchdata.shape[0]):
        if central_area[i] == 0:
            indices_central = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).cuda()
            indices_boundary = torch.tensor([8, 9, 10, 11, 12, 13, 14, 15]).cuda()
        elif central_area[i] == 1:
            indices_central = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]).cuda()
            indices_boundary = torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]).cuda()
        elif central_area[i] == 2:
            indices_central = torch.tensor([8, 9, 10, 11, 12, 13, 14, 15]).cuda()
            indices_boundary = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).cuda()
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
        # 3 * 224 * 448 -> 3  * 3136 * 32
        patchselect_central = rearrange(patchselect_central, 'c (h p_h) (w p_w)-> c (p_h p_w) (h w) ', h=4, w=8)

        datacentral_list.append(patchselect_central.unsqueeze(0))
        databoundary_list.append(patchselect_boundary.unsqueeze(0))

        # b * 3 * 12544 * 8
    batchdata_boundary = torch.cat(databoundary_list, dim=0)
    # b * 3 * 3136 * 32
    batchdata_central = torch.cat(datacentral_list, dim=0)

    return batchdata_central ,batchdata_boundary ,central_area


class aldlayer4decv8(nn.Module):
    def __init__(self, num_train):
        super(aldlayer4decv8, self).__init__()
        # two-step version for aircraft with decv8
        self.params_centre = Parameter(torch.Tensor(num_train, 32, 32), requires_grad=True)
        self.params_global = Parameter(torch.Tensor(num_train, 16, 16), requires_grad=True)
        torch.nn.init.normal_(self.params_centre)
        torch.nn.init.normal_(self.params_global)

    def forward(self, x, batch_seq, batchsize):

        x_centre, x_boundary ,_ = faster_rcnn_detection(x,5,batch_seq)

        if x_boundary.shape[0] == batchsize:
            batchnum = batchsize
        else:
            batchnum = x_boundary.shape[0]

        start = batch_seq * batchsize
        end = batch_seq * batchsize + batchnum

        first = True
        for i in range(start, end):
            rcm = self.params_centre[i]
            rcm = F.softmax(rcm,dim=0)
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
                result_central = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic, 0)
                result_central = torch.cat((result_single_pic, result_central), 0)

        first = True
        for i in range(start, end):
            rcm = self.params_global[i]
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
            result_r = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 +
                        b11 + b12 + b13 + b14 + b15 + b16)
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_global = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic, 0)
                result_global = torch.cat((result_single_pic, result_global), 0)


        result_central = result_central.to(torch.float32)
        result_global = result_global.to(torch.float32)



        x_mat_central = x_centre.matmul(result_central)
        # x_mat_central = x_centre
        x_global = self.datarest(x_mat_central, x_boundary)
        x_mat_global = x_global.matmul(result_global)
        # x_mat_global = x_global
        # b * 3  * 12544 * 16 ->  b * 3 * 448 * 448
        x_mat_global = rearrange(x_mat_global, 'b c (p_h p_w) (h w) -> b c (h p_h) (w p_w)', h=4, w=4, p_h=112,
                                 p_w=112)
        return x_mat_global, result_central, result_global

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

    def datarest(self, batchdata_centre, batchdata_boundary):
        # batchdata_centre 转化成同一尺寸   batchdata8 b * 3 * 3136 * 32  -> b * 3  *( 112 * 112) * 8
        # 先复原 在分割
        # b * 3 * 3136 * 32  -> b * 3  * 32 * 56 * 56
        batchdata_centre = rearrange(batchdata_centre, 'b c (p_h p_w) l  -> b c l p_h p_w ', p_h=56, p_w=56)
        # b * 3 * 32 * 56 * 56 -> b * 3 * 224 * 448
        batchdata_centre = rearrange(batchdata_centre, 'b c (h w) p_h p_w -> b c (h p_h) (w p_w) ', h=4, w=8)
        # b * 3 * 224 * 448 -> b * 3 * (112 * 112) * 8
        batchdata_centre = rearrange(batchdata_centre, 'b c (h p_h) (w p_w) -> b c (p_h p_w) (h w)', h=2, w=4)
        batchdata = torch.cat((batchdata_boundary, batchdata_centre), dim=3)
        # index = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7]
        # batchdata = batchdata[:, :, :, index]
        return batchdata


# 不用 two-step的方式做 mix-scale v1  同时使用检测模块来检测中心区域
class  ald_decv1(nn.Module):

    def __init__(self,num_train):
        super(ald_decv1, self).__init__()
        self.params_boundary = Parameter(torch.Tensor(num_train,12,12), requires_grad=True)
        self.params_centre = Parameter(torch.Tensor(num_train,16,16), requires_grad=True)
        torch.nn.init.normal_(self.params_boundary)
        torch.nn.init.normal_(self.params_centre)


    def forward(self, x,batch_seq,batchsize):
        x_centre, x_boundary, centre_area = self.faster_rcnn_detection(x, 3)

        if x_centre.shape[0] == batchsize :
            batchnum = batchsize
        else:
            batchnum = x_centre.shape[0]

        start = batch_seq*batchsize
        end   = batch_seq*batchsize + batchnum

        first = True
        for i in range(start,end):
            rcm = self.params_boundary[i]
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

            result_r = (b1 + b2 + b3 + b4  + b5 + b6 + b7 + b8 + b9 + b10 +
                      b11 + b12 )
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_boundary = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic,0)
                result_boundary = torch.cat((result_single_pic,result_boundary),0)

        first = True
        for i in range(start,end):
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
            result_r = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 +
                      b11 + b12 + b13 + b14 + b15 + b16)
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_centre = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic,0)
                result_centre = torch.cat((result_single_pic,result_centre),0)

        result_boundary = result_boundary.to(torch.float32)
        result_centre = result_centre.to(torch.float32)

        x_mat_centre = x_centre.matmul(result_centre)
        x_mat_boundary = x_boundary.matmul(result_boundary)

        x_ouput = self.datareset(x_mat_centre,x_mat_boundary,centre_area)
        return x_ouput,result_centre,result_boundary
    def getrank(self,rcm):
        e = 0.0000001
        # print(rcm.shape)
        size = rcm.shape[1]
        c = torch.full((size, size), -100000.0)
        c_cuda = c
        if torch.cuda.is_available() :
            # c_cuda = c.to(self.device)
            c_cuda = c.cuda()
        disturb = torch.from_numpy(np.random.normal(0, 0.000001, (size, size)))
        disturb_cuda = disturb.cuda()
        rcm = rcm + disturb_cuda
        maxvalue = torch.max(rcm).detach()
        rcm_flatten = torch.squeeze(rcm.detach().reshape(1,-1),0)
        max2, _ = torch.topk(rcm_flatten, 2)
        sec_maxvalue = max2[1]
        if (maxvalue == sec_maxvalue):
            print("-----------------------")
            print('rcm',rcm)
            print('max',maxvalue)
            print('sec',sec_maxvalue)
            print("-----------------------")
        b = torch.relu(rcm - (maxvalue + sec_maxvalue) /2 ) / ((maxvalue - sec_maxvalue) /2 )
        b_max = torch.max(b).detach()
        b = b/ b_max
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
        assert max_b_value == 1.0 , "maxvalue of  b not 1 is : " + str(max_b_value)
        # if(max_b_value != 1.0):
        #     print(b)
        b_value = b_value.squeeze(0)
        b_value = b_value.to(torch.float32)
        cmul = torch.mm(b_value, c_cuda) + torch.mm(c_cuda, b_value)
        cmul = cmul.unsqueeze(0)
        rcmget = rcm + cmul
        return b , rcmget
    def minmaxscaler(self,rcm):
        max = torch.max(rcm)
        min = torch.min(rcm)

        return (rcm.data-min) /(max-min)

    def faster_rcnn_detection(self,batchdata, obj_target):
        # 加载pytorch自带的预训练Faster RCNN目标检测模型  obj_target 代表要检测出的物体类别 5 代表 aircraft
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.cuda()
        model.eval()

        # 将图像输入神经网络模型中，得到输出
        output = model(batchdata)
        # 分出中心和边界区域
        datacentral_list = []
        databoundary_list = []
        central_area = []
        # 算出中心区域的index
        for i in range(len((output))):
            labels = output[i]['labels'].cpu().detach().numpy()  # 预测每一个obj的标签
            bboxes = output[i]['boxes'].cpu().detach().numpy()  # 预测每一个obj的边框
            declabel = np.argwhere(labels == obj_target)
            # 通过目标检测来选取中心区域
            if declabel.size != 0:
                index = declabel[0][0]
                x1, y1, x2, y2 = bboxes[index]
                y1 = int(y1)
                y2 = int(y2)
                x1 = int(x1)
                x2 = int(x2)
                # 从上到下 从左到右 一共九个候选区域 选择一个目标区域所在位置最大的
                # 三个区域是 0-224 112-336 224-448
                maxarea = float('-inf')
                y = np.zeros((448,448), dtype=int)
                y[x1:x2,y1:y2] = 1
                for i in range(3):
                    for j in range(3):
                        w_left = j * 112
                        w_right = (j + 2) * 112
                        h_up = i * 112
                        h_down = (i + 2) * 112
                        # print(sum(y[w_left:w_right, h_up:h_down]))
                        area = np.sum(y[w_left:w_right,h_up:h_down])
                        # print(area)
                        if (maxarea < area):
                            maxarea = area
                            maxindex = i * 3 + j
                # print(maxarea)
                # print(maxindex)
                central_area.append(maxindex)
            else:
                central_area.append(4)
        for i in range(batchdata.shape[0]):
            if central_area[i] == 0:
                indices_central = torch.tensor([0, 1, 4, 5]).cuda()
                indices_boundary = torch.tensor([2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).cuda()
            elif central_area[i] == 1:
                indices_central = torch.tensor([1, 2, 5, 6]).cuda()
                indices_boundary = torch.tensor([0, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]).cuda()
            elif central_area[i] == 2:
                indices_central = torch.tensor([2, 3, 6, 7]).cuda()
                indices_boundary = torch.tensor([0, 1, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15]).cuda()
            elif central_area[i] == 3:
                indices_central = torch.tensor([4, 5, 8, 9]).cuda()
                indices_boundary = torch.tensor([0, 1, 2, 3, 6, 7, 10, 11, 12, 13, 14, 15]).cuda()
            elif central_area[i] == 4:
                indices_central = torch.tensor([5, 6, 9, 10]).cuda()
                indices_boundary = torch.tensor([0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]).cuda()
            elif central_area[i] == 5:
                indices_central = torch.tensor([6, 7, 10, 11]).cuda()
                indices_boundary = torch.tensor([0, 1, 2, 3, 4, 5, 8, 9, 12, 13, 14, 15]).cuda()
            elif central_area[i] == 6:
                indices_central = torch.tensor([8, 9, 12, 13]).cuda()
                indices_boundary = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15]).cuda()
            elif central_area[i] == 7:
                indices_central = torch.tensor([9, 10, 13, 14]).cuda()
                indices_boundary = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 15]).cuda()
            elif central_area[i] == 8:
                indices_central = torch.tensor([10, 11, 14, 15]).cuda()
                indices_boundary = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13]).cuda()
            # indices_central = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]).cuda()
            # indices_boundary = torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]).cuda()
            batchdata_cur = batchdata[i]
            batchdata_cur = rearrange(batchdata_cur, 'c (h p_h) (w p_w)  -> c  (p_h p_w) (h w) ', p_h=112, p_w=112)
            patchselect_boundary = torch.index_select(batchdata_cur, 2, indices_boundary)
            patchselect_central = torch.index_select(batchdata_cur, 2, indices_central)
            # 对中心区域先复原在分割
            # 3  * 12544  * 4 -> 3 * 224 * 224
            patchselect_central = rearrange(patchselect_central, 'c (p_h p_w) (h w) -> c (h p_h) (w p_w)', h=2, w=2,
                                            p_h=112, p_w=112)
            # 3 * 224 * 224 -> 3  * 3136 * 16
            patchselect_central = rearrange(patchselect_central, 'c (h p_h) (w p_w)-> c (p_h p_w) (h w) ', h=4, w=4)

            datacentral_list.append(patchselect_central.unsqueeze(0))
            databoundary_list.append(patchselect_boundary.unsqueeze(0))

            # b * 3 * 12544 * 12
        batchdata_boundary = torch.cat(databoundary_list, dim=0)
        # b * 3 * 3136 * 16
        batchdata_central = torch.cat(datacentral_list, dim=0)

        return batchdata_central, batchdata_boundary, central_area

    def datareset(self, batchdata_centre, batchdata_boundary,central_area):
        # batchdata_centre 转化成同一尺寸   batchdata8 b * 3 * 3136 * 16  -> b * 3  *( 112 * 112) * 4
        # 先复原 在分割
        # b * 3 * 3136 * 16  -> b * 3  * 16 * 56 * 56
        batchdata_centre = rearrange(batchdata_centre, 'b c (p_h p_w) l  -> b c l p_h p_w ', p_h=56, p_w=56)
        # b * 3 * 16 * 56 * 56 -> b * 3 * 224 * 224
        batchdata_centre = rearrange(batchdata_centre, 'b c (h w) p_h p_w -> b c (h p_h) (w p_w) ', h=4, w=4)
        # b * 3 * 224 * 224 -> b * 3 * (112 * 112) * 4
        batchdata_centre = rearrange(batchdata_centre, 'b c (h p_h) (w p_w) -> b c (p_h p_w) (h w)', h=2, w=2)
        batchdata = torch.cat((batchdata_boundary, batchdata_centre), dim=3)
        # index = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7]
        # batchdata = batchdata[:, :, :, index]
        batchsize = batchdata.shape[0]
        indexdict = {0: [12,13,0,1,14,15,2,3,4,5,6,7,8,9,10,11],
                     1: [0,12,13,1,2,14,15,3,4,5,6,7,8,9,10,11],
                     2: [0,1,12,13,2,3,14,15,4,5,6,7,8,9,10,11],
                     3: [0,1,2,3,12,13,4,5,14,15,6,7,8,9,10,11],
                     4: [0,1,2,3,4,12,13,5,6,14,15,7,8,9,10,11],
                     5: [0,1,2,3,4,5,12,13,6,7,14,15,8,9,10,11],
                     6: [0,1,2,3,4,5,6,7,12,13,8,9,14,15,10,11],
                     7: [0,1,2,3,4,5,6,7,8,12,13,9,10,14,15,11],
                     8: [0,1,2,3,4,5,6,7,8,9,12,13,10,11,14,15]
                     }
        for i in range(batchsize):
            index = indexdict[central_area[i]]
            batchdata[i] = batchdata[i, :, :, index]

        x_out = rearrange(batchdata, 'b c (p_h p_w) (h w) -> b c (h p_h) (w p_w)', h=4, w=4, p_h=112, p_w=112)
        return x_out


# 不用 two-step的方式做 mix-scale v1  同时使用特征提取的方式确定中心区域
class  ald_fev1(nn.Module):

    def __init__(self,num_train):
        super(ald_fev1, self).__init__()
        self.params_boundary = Parameter(torch.Tensor(num_train,12,12), requires_grad=True)
        self.params_centre = Parameter(torch.Tensor(num_train,16,16), requires_grad=True)
        self.avgpool1 = nn.AdaptiveAvgPool2d(4)
        torch.nn.init.normal_(self.params_boundary)
        torch.nn.init.normal_(self.params_centre)


    def forward(self, x,batch_seq,batchsize,feature_extractor):
        x_centre, x_boundary, select_index = self.fea_extarea(x,feature_extractor,batch_seq)

        if x_centre.shape[0] == batchsize :
            batchnum = batchsize
        else:
            batchnum = x_centre.shape[0]

        start = batch_seq*batchsize
        end   = batch_seq*batchsize + batchnum

        first = True
        for i in range(start,end):
            rcm = self.params_boundary[i]
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

            result_r = (b1 + b2 + b3 + b4  + b5 + b6 + b7 + b8 + b9 + b10 +
                      b11 + b12 )
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_boundary = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic,0)
                result_boundary = torch.cat((result_single_pic,result_boundary),0)

        first = True
        for i in range(start,end):
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
            result_r = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 +
                      b11 + b12 + b13 + b14 + b15 + b16)
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_centre = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic,0)
                result_centre = torch.cat((result_single_pic,result_centre),0)

        result_boundary = result_boundary.to(torch.float32)
        result_centre = result_centre.to(torch.float32)

        x_mat_centre = x_centre.matmul(result_centre)
        x_mat_boundary = x_boundary.matmul(result_boundary)

        x_ouput = self.datareset(x_mat_centre,x_mat_boundary,select_index)
        return x_ouput,result_centre,result_boundary ,
    def getrank(self,rcm):
        e = 0.0000001
        # print(rcm.shape)
        size = rcm.shape[1]
        c = torch.full((size, size), -100000.0)
        c_cuda = c
        if torch.cuda.is_available() :
            # c_cuda = c.to(self.device)
            c_cuda = c.cuda()
        disturb = torch.from_numpy(np.random.normal(0, 0.000001, (size, size)))
        disturb_cuda = disturb.cuda()
        rcm = rcm + disturb_cuda
        maxvalue = torch.max(rcm).detach()
        rcm_flatten = torch.squeeze(rcm.detach().reshape(1,-1),0)
        max2, _ = torch.topk(rcm_flatten, 2)
        sec_maxvalue = max2[1]
        if (maxvalue == sec_maxvalue):
            print("-----------------------")
            print('rcm',rcm)
            print('max',maxvalue)
            print('sec',sec_maxvalue)
            print("-----------------------")
        b = torch.relu(rcm - (maxvalue + sec_maxvalue) /2 ) / ((maxvalue - sec_maxvalue) /2 )
        b_max = torch.max(b).detach()
        b = b/ b_max
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
        assert max_b_value == 1.0 , "maxvalue of  b not 1 is : " + str(max_b_value)
        # if(max_b_value != 1.0):
        #     print(b)
        b_value = b_value.squeeze(0)
        b_value = b_value.to(torch.float32)
        cmul = torch.mm(b_value, c_cuda) + torch.mm(c_cuda, b_value)
        cmul = cmul.unsqueeze(0)
        rcmget = rcm + cmul
        return b , rcmget
    def minmaxscaler(self,rcm):
        max = torch.max(rcm)
        min = torch.min(rcm)

        return (rcm.data-min) /(max-min)
    def fea_extarea(self,batchdata,feature_extractor,batch_seq):
        _, _, _, _ ,fea_map = feature_extractor(batchdata)

        avgout = torch.mean(fea_map, dim=1, keepdim=True)
        self.tensor2img(avgout[0], 'featuremap' + str(batch_seq))
        avgout = self.avgpool1(avgout)
        avgout = avgout.flatten(1)
        _ ,selectindex = torch.topk(avgout,16,dim=1)
        datacentral_list = []
        databoundary_list = []
        for i in range(batchdata.shape[0]):

            indices_central = selectindex[i,0:4].cuda()
            centre_list = selectindex[i, 0:4].tolist()
            all_list = [i for i in range(16)]
            boundary_list = [i for i in all_list if i not in centre_list]
            indices_boundary = torch.tensor(boundary_list).cuda()
            # indices_central = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]).cuda()
            # indices_boundary = torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]).cuda()
            batchdata_cur = batchdata[i]
            batchdata_cur = rearrange(batchdata_cur, 'c (h p_h) (w p_w)  -> c  (p_h p_w) (h w) ', p_h=112, p_w=112)
            patchselect_boundary = torch.index_select(batchdata_cur, 2, indices_boundary)
            patchselect_central = torch.index_select(batchdata_cur, 2, indices_central)
            # 对中心区域先复原在分割
            # 3  * 12544  * 4 -> 3 * 224 * 224
            patchselect_central = rearrange(patchselect_central, 'c (p_h p_w) (h w) -> c (h p_h) (w p_w)', h=2, w=2,
                                            p_h=112, p_w=112)
            # 3 * 224 * 224 -> 3  * 3136 * 16
            patchselect_central = rearrange(patchselect_central, 'c (h p_h) (w p_w)-> c (p_h p_w) (h w) ', h=4, w=4)

            datacentral_list.append(patchselect_central.unsqueeze(0))
            databoundary_list.append(patchselect_boundary.unsqueeze(0))
        # b * 3 * 12544 * 12
        batchdata_boundary = torch.cat(databoundary_list, dim=0)
        # b * 3 * 3136 * 16
        batchdata_central = torch.cat(datacentral_list, dim=0)
        return batchdata_central, batchdata_boundary ,selectindex
    def datareset(self, batchdata_centre, batchdata_boundary,central_area):
        # batchdata_centre 转化成同一尺寸   batchdata8 b * 3 * 3136 * 16  -> b * 3  *( 112 * 112) * 4
        # 先复原 在分割
        # b * 3 * 3136 * 16  -> b * 3  * 16 * 56 * 56
        batchdata_centre = rearrange(batchdata_centre, 'b c (p_h p_w) l  -> b c l p_h p_w ', p_h=56, p_w=56)
        # b * 3 * 16 * 56 * 56 -> b * 3 * 224 * 224
        batchdata_centre = rearrange(batchdata_centre, 'b c (h w) p_h p_w -> b c (h p_h) (w p_w) ', h=4, w=4)
        # b * 3 * 224 * 224 -> b * 3 * (112 * 112) * 4
        batchdata_centre = rearrange(batchdata_centre, 'b c (h p_h) (w p_w) -> b c (p_h p_w) (h w)', h=2, w=2)
        batchdata = torch.cat((batchdata_centre, batchdata_boundary), dim=3)
        # index = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7]
        # batchdata = batchdata[:, :, :, index]
        batchsize = batchdata.shape[0]
        # 可以优化
        for i in range(batchsize):
            index = [ -1 for i in range(16)]
            for j in range(4):
                index[central_area[i,j]] = j
            start = 4
            for k in range(16):
                if index[k] == -1:
                    index[k] = start
                    start = start + 1
            # index = indexdict[central_area[i]]
            batchdata[i] = batchdata[i, :, :, index]

        x_out = rearrange(batchdata, 'b c (p_h p_w) (h w) -> b c (h p_h) (w p_w)', h=4, w=4, p_h=112, p_w=112)
        return x_out
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


# 不用 two-step的方式做 mix-scale v1  同时使用特征提取的方式确定中心区域 并且只在中心区域上学习打乱
class  ald_fev1_onlycentre(nn.Module):

    def __init__(self,num_train):
        super(ald_fev1_onlycentre, self).__init__()
        # self.params_boundary = Parameter(torch.Tensor(num_train,12,12), requires_grad=True)
        self.params_centre = Parameter(torch.Tensor(num_train,16,16), requires_grad=True)
        self.avgpool1 = nn.AdaptiveAvgPool2d(4)
        # torch.nn.init.normal_(self.params_boundary)
        torch.nn.init.normal_(self.params_centre)


    def forward(self, x,batch_seq,batchsize,feature_extractor):
        x_centre, x_boundary, select_index = self.fea_extarea(x,feature_extractor,batch_seq)

        if x_centre.shape[0] == batchsize :
            batchnum = batchsize
        else:
            batchnum = x_centre.shape[0]

        start = batch_seq*batchsize
        end   = batch_seq*batchsize + batchnum

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
        for i in range(start,end):
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
            result_r = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 +
                      b11 + b12 + b13 + b14 + b15 + b16)
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_centre = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic,0)
                result_centre = torch.cat((result_single_pic,result_centre),0)

        # result_boundary = result_boundary.to(torch.float32)
        result_centre = result_centre.to(torch.float32)

        x_mat_centre = x_centre.matmul(result_centre)
        # x_mat_boundary = x_boundary.matmul(result_boundary)

        x_ouput = self.datareset(x_mat_centre,x_boundary,select_index)
        return x_ouput,result_centre,select_index
    def getrank(self,rcm):
        e = 0.0000001
        # print(rcm.shape)
        size = rcm.shape[1]
        c = torch.full((size, size), -100000.0)
        c_cuda = c
        if torch.cuda.is_available() :
            # c_cuda = c.to(self.device)
            c_cuda = c.cuda()
        disturb = torch.from_numpy(np.random.normal(0, 0.000001, (size, size)))
        disturb_cuda = disturb.cuda()
        rcm = rcm + disturb_cuda
        maxvalue = torch.max(rcm).detach()
        rcm_flatten = torch.squeeze(rcm.detach().reshape(1,-1),0)
        max2, _ = torch.topk(rcm_flatten, 2)
        sec_maxvalue = max2[1]
        if (maxvalue == sec_maxvalue):
            print("-----------------------")
            print('rcm',rcm)
            print('max',maxvalue)
            print('sec',sec_maxvalue)
            print("-----------------------")
        b = torch.relu(rcm - (maxvalue + sec_maxvalue) /2 ) / ((maxvalue - sec_maxvalue) /2 )
        b_max = torch.max(b).detach()
        b = b/ b_max
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
        assert max_b_value == 1.0 , "maxvalue of  b not 1 is : " + str(max_b_value)
        # if(max_b_value != 1.0):
        #     print(b)
        b_value = b_value.squeeze(0)
        b_value = b_value.to(torch.float32)
        cmul = torch.mm(b_value, c_cuda) + torch.mm(c_cuda, b_value)
        cmul = cmul.unsqueeze(0)
        rcmget = rcm + cmul
        return b , rcmget
    def minmaxscaler(self,rcm):
        max = torch.max(rcm)
        min = torch.min(rcm)

        return (rcm.data-min) /(max-min)
    def fea_extarea(self,batchdata,feature_extractor,batch_seq):
        _, _, _, _ ,fea_map = feature_extractor(batchdata)

        avgout = torch.mean(fea_map, dim=1, keepdim=True)
        # self.tensor2img(avgout[0], 'featuremap' + str(batch_seq))
        avgout = self.avgpool1(avgout)
        avgout = avgout.flatten(1)
        _ ,selectindex = torch.topk(avgout,16,dim=1)
        datacentral_list = []
        databoundary_list = []
        for i in range(batchdata.shape[0]):

            indices_central = selectindex[i,0:4].cuda()
            centre_list = selectindex[i, 0:4].tolist()
            all_list = [i for i in range(16)]
            boundary_list = [i for i in all_list if i not in centre_list]
            # random.shuffle(boundary_list)
            indices_boundary = torch.tensor(boundary_list).cuda()
            # indices_central = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]).cuda()
            # indices_boundary = torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]).cuda()
            batchdata_cur = batchdata[i]
            batchdata_cur = rearrange(batchdata_cur, 'c (h p_h) (w p_w)  -> c  (p_h p_w) (h w) ', p_h=112, p_w=112)
            patchselect_boundary = torch.index_select(batchdata_cur, 2, indices_boundary)
            patchselect_central = torch.index_select(batchdata_cur, 2, indices_central)
            # 对中心区域先复原在分割
            # 3  * 12544  * 4 -> 3 * 224 * 224
            patchselect_central = rearrange(patchselect_central, 'c (p_h p_w) (h w) -> c (h p_h) (w p_w)', h=2, w=2,
                                            p_h=112, p_w=112)
            # 3 * 224 * 224 -> 3  * 3136 * 16
            patchselect_central = rearrange(patchselect_central, 'c (h p_h) (w p_w)-> c (p_h p_w) (h w) ', h=4, w=4)

            datacentral_list.append(patchselect_central.unsqueeze(0))
            databoundary_list.append(patchselect_boundary.unsqueeze(0))
        # b * 3 * 12544 * 12
        batchdata_boundary = torch.cat(databoundary_list, dim=0)
        # b * 3 * 3136 * 16
        batchdata_central = torch.cat(datacentral_list, dim=0)
        return batchdata_central, batchdata_boundary ,selectindex
    def datareset(self, batchdata_centre, batchdata_boundary,central_area):
        # batchdata_centre 转化成同一尺寸   batchdata8 b * 3 * 3136 * 16  -> b * 3  *( 112 * 112) * 4
        # 先复原 在分割
        # b * 3 * 3136 * 16  -> b * 3  * 16 * 56 * 56
        batchdata_centre = rearrange(batchdata_centre, 'b c (p_h p_w) l  -> b c l p_h p_w ', p_h=56, p_w=56)
        # b * 3 * 16 * 56 * 56 -> b * 3 * 224 * 224
        batchdata_centre = rearrange(batchdata_centre, 'b c (h w) p_h p_w -> b c (h p_h) (w p_w) ', h=4, w=4)
        # b * 3 * 224 * 224 -> b * 3 * (112 * 112) * 4
        batchdata_centre = rearrange(batchdata_centre, 'b c (h p_h) (w p_w) -> b c (p_h p_w) (h w)', h=2, w=2)
        batchdata = torch.cat((batchdata_centre, batchdata_boundary), dim=3)
        # index = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7]
        # batchdata = batchdata[:, :, :, index]
        batchsize = batchdata.shape[0]

        # 可以优化
        for i in range(batchsize):
            index = [ -1 for i in range(16)]
            for j in range(4):
                index[central_area[i,j]] = j
            start = 4
            for k in range(16):
                if index[k] == -1:
                    index[k] = start
                    start = start + 1
            # index = indexdict[central_area[i]]
            batchdata[i] = batchdata[i, :, :, index]

        x_out = rearrange(batchdata, 'b c (p_h p_w) (h w) -> b c (h p_h) (w p_w)', h=4, w=4, p_h=112, p_w=112)
        return x_out
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

# 不用 two-step的方式 在aircreaft 做 mix-scale v8  同时使用特征提取的方式确定中心区域 并且只在中心区域上学习打乱
class  ald_fev8_onlycentre(nn.Module):

    def __init__(self,num_train):
        super(ald_fev8_onlycentre, self).__init__()
        # self.params_boundary = Parameter(torch.Tensor(num_train,12,12), requires_grad=True)
        self.params_centre = Parameter(torch.Tensor(num_train,32,32), requires_grad=True)
        self.avgpool1 = nn.AdaptiveAvgPool2d(4)
        # torch.nn.init.normal_(self.params_boundary)
        torch.nn.init.normal_(self.params_centre)


    def forward(self, x,batch_seq,batchsize,feature_extractor):
        x_centre, x_boundary, select_index = self.fea_extarea(x,feature_extractor,batch_seq)

        if x_centre.shape[0] == batchsize :
            batchnum = batchsize
        else:
            batchnum = x_centre.shape[0]

        start = batch_seq*batchsize
        end   = batch_seq*batchsize + batchnum

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
        for i in range(start,end):
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
                      b11 + b12 + b13 + b14 + b15 + b16  + b17 + b18 + b19 + b20 +
                      b21 + b22 + b23 + b24 + b25 + b26 + b27 + b28 + b29 + b30 + b31 + b32 )
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_centre = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic,0)
                result_centre = torch.cat((result_single_pic,result_centre),0)

        # result_boundary = result_boundary.to(torch.float32)
        result_centre = result_centre.to(torch.float32)

        x_mat_centre = x_centre.matmul(result_centre)
        # x_mat_boundary = x_boundary.matmul(result_boundary)

        x_ouput = self.datareset(x_mat_centre,x_boundary,select_index)
        return x_ouput,result_centre,select_index
    def getrank(self,rcm):
        e = 0.0000001
        # print(rcm.shape)
        size = rcm.shape[1]
        c = torch.full((size, size), -100000.0)
        c_cuda = c
        if torch.cuda.is_available() :
            # c_cuda = c.to(self.device)
            c_cuda = c.cuda()
        disturb = torch.from_numpy(np.random.normal(0, 0.000001, (size, size)))
        disturb_cuda = disturb.cuda()
        rcm = rcm + disturb_cuda
        maxvalue = torch.max(rcm).detach()
        rcm_flatten = torch.squeeze(rcm.detach().reshape(1,-1),0)
        max2, _ = torch.topk(rcm_flatten, 2)
        sec_maxvalue = max2[1]
        if (maxvalue == sec_maxvalue):
            print("-----------------------")
            print('rcm',rcm)
            print('max',maxvalue)
            print('sec',sec_maxvalue)
            print("-----------------------")
        b = torch.relu(rcm - (maxvalue + sec_maxvalue) /2 ) / ((maxvalue - sec_maxvalue) /2 )
        b_max = torch.max(b).detach()
        b = b/ b_max
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
        assert max_b_value == 1.0 , "maxvalue of  b not 1 is : " + str(max_b_value)
        # if(max_b_value != 1.0):
        #     print(b)
        b_value = b_value.squeeze(0)
        b_value = b_value.to(torch.float32)
        cmul = torch.mm(b_value, c_cuda) + torch.mm(c_cuda, b_value)
        cmul = cmul.unsqueeze(0)
        rcmget = rcm + cmul
        return b , rcmget
    def minmaxscaler(self,rcm):
        max = torch.max(rcm)
        min = torch.min(rcm)

        return (rcm.data-min) /(max-min)
    def fea_extarea(self,batchdata,feature_extractor,batch_seq):
        _, _, _, _ ,fea_map = feature_extractor(batchdata)

        avgout = torch.mean(fea_map, dim=1, keepdim=True)
        # self.tensor2img(avgout[0], 'featuremap' + str(batch_seq))
        avgout = self.avgpool1(avgout)
        avgout = avgout.flatten(1)
        _ ,selectindex = torch.topk(avgout,16,dim=1)
        datacentral_list = []
        databoundary_list = []
        for i in range(batchdata.shape[0]):

            indices_central = selectindex[i,0:8].cuda()
            centre_list = selectindex[i, 0:8].tolist()
            all_list = [i for i in range(16)]
            boundary_list = [i for i in all_list if i not in centre_list]
            # random.shuffle(boundary_list)
            indices_boundary = torch.tensor(boundary_list).cuda()
            # indices_central = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]).cuda()
            # indices_boundary = torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]).cuda()
            batchdata_cur = batchdata[i]
            # 3 * 448 * 448 -> 3 * (12544) * 16
            batchdata_cur = rearrange(batchdata_cur, 'c (h p_h) (w p_w)  -> c  (p_h p_w) (h w) ', p_h=112, p_w=112)
            patchselect_boundary = torch.index_select(batchdata_cur, 2, indices_boundary)
            patchselect_central = torch.index_select(batchdata_cur, 2, indices_central)
            # 对中心区域先复原在分割
            # 3  * 12544  * 8 -> 3 * 224 * 448
            patchselect_central = rearrange(patchselect_central, 'c (p_h p_w) (h w) -> c (h p_h) (w p_w)', h=2, w=4,
                                            p_h=112, p_w=112)
            # 3 * 224 * 448 -> 3  * 3136 * 32
            patchselect_central = rearrange(patchselect_central, 'c (h p_h) (w p_w)-> c (p_h p_w) (h w) ', h=4, w=8)

            datacentral_list.append(patchselect_central.unsqueeze(0))
            databoundary_list.append(patchselect_boundary.unsqueeze(0))
        # b * 3 * 12544 * 8
        batchdata_boundary = torch.cat(databoundary_list, dim=0)
        # b * 3 * 3136 * 32
        batchdata_central = torch.cat(datacentral_list, dim=0)
        return batchdata_central, batchdata_boundary ,selectindex
    def datareset(self, batchdata_centre, batchdata_boundary,central_area):
        # batchdata_centre 转化成同一尺寸   batchdata8 b * 3 * 3136 * 32  -> b * 3  *( 112 * 112) * 8
        # 先复原 在分割
        # b * 3 * 3136 * 32  -> b * 3  * 32 * 56 * 56
        batchdata_centre = rearrange(batchdata_centre, 'b c (p_h p_w) l  -> b c l p_h p_w ', p_h=56, p_w=56)
        # b * 3 * 32 * 56 * 56 -> b * 3 * 224 * 448
        batchdata_centre = rearrange(batchdata_centre, 'b c (h w) p_h p_w -> b c (h p_h) (w p_w) ', h=4, w=8)
        # b * 3 * 224 * 448 -> b * 3 * (112 * 112) * 8
        batchdata_centre = rearrange(batchdata_centre, 'b c (h p_h) (w p_w) -> b c (p_h p_w) (h w)', h=2, w=4)
        batchdata = torch.cat((batchdata_centre, batchdata_boundary), dim=3)
        # index = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7]
        # batchdata = batchdata[:, :, :, index]
        batchsize = batchdata.shape[0]
        # 可以优化
        for i in range(batchsize):
            index = [ -1 for i in range(16)]
            for j in range(8):
                index[central_area[i,j]] = j
            start = 8
            for k in range(16):
                if index[k] == -1:
                    index[k] = start
                    start = start + 1
            # index = indexdict[central_area[i]]
            batchdata[i] = batchdata[i, :, :, index]

        x_out = rearrange(batchdata, 'b c (p_h p_w) (h w) -> b c (h p_h) (w p_w)', h=4, w=4, p_h=112, p_w=112)
        return x_out
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

# 用 two-step的方式做 mix-scale v1  同时使用特征提取的方式确定中心区域 并把中心区域分为两部分
# 一共选择8块 一部分用小的粒度 一部分用大的粒度
class  ald_fev1_mixsca(nn.Module):

    def __init__(self,num_train):
        super(ald_fev1_mixsca, self).__init__()
        # self.params_boundary = Parameter(torch.Tensor(num_train,12,12), requires_grad=True)
        self.params_centre = Parameter(torch.Tensor(num_train,16,16), requires_grad=True)
        self.params_global = Parameter(torch.Tensor(num_train, 8, 8), requires_grad=True)
        self.avgpool1 = nn.AdaptiveAvgPool2d(4)
        # torch.nn.init.normal_(self.params_boundary)
        torch.nn.init.normal_(self.params_centre)


    def forward(self, x,batch_seq,batchsize,feature_extractor):
        x_centre_small, x_centre_large , x_boundary, select_index = self.fea_extarea(x,feature_extractor,batch_seq)

        if x_centre_small.shape[0] == batchsize :
            batchnum = batchsize
        else:
            batchnum = x_centre_small.shape[0]

        start = batch_seq*batchsize
        end   = batch_seq*batchsize + batchnum



        first = True
        for i in range(start,end):
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
            result_r = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 +
                      b11 + b12 + b13 + b14 + b15 + b16)
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_centre = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic,0)
                result_centre = torch.cat((result_single_pic,result_centre),0)

        first = True
        for i in range(start, end):
            rcm = self.params_global[i]
            rcm = F.softmax(rcm, dim=0)
            b1, rcm1 = self.getrank(rcm)
            b2, rcm2 = self.getrank(rcm1)
            b3, rcm3 = self.getrank(rcm2)
            b4, rcm4 = self.getrank(rcm3)
            b5, rcm5 = self.getrank(rcm4)
            b6, rcm6 = self.getrank(rcm5)
            b7, rcm7 = self.getrank(rcm6)
            b8, rcm8 = self.getrank(rcm7)
            result_r = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 )
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_global = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic, 0)
                result_global = torch.cat((result_single_pic, result_global), 0)


        # result_boundary = result_boundary.to(torch.float32)
        result_global = result_global.to(torch.float32)

        result_centre = result_centre.to(torch.float32)
        x_mat_centre_small = x_centre_small.matmul(result_centre)
        # b * 3 * 3136 * 16 -> b * 3 * 12544 * 4
        # b * 3 * 3136 * 16  -> b * 3  * 16 * 56 * 56
        x_mat_centre_small = rearrange(x_mat_centre_small, 'b c (p_h p_w) l  -> b c l p_h p_w ', p_h=56, p_w=56)
        # b * 3 * 16 * 56 * 56 -> b * 3 * 224 * 224
        x_mat_centre_small = rearrange(x_mat_centre_small, 'b c (h w) p_h p_w -> b c (h p_h) (w p_w) ', h=4, w=4)
        # b * 3 * 224 * 224 -> b * 3 * (112 * 112) * 4
        x_mat_centre_small = rearrange(x_mat_centre_small, 'b c (h p_h) (w p_w) -> b c (p_h p_w) (h w)', h=2, w=2)

        x_centre = torch.cat((x_mat_centre_small,x_centre_large),dim=3)
        x_mat_centre =  x_centre.matmul(result_global)
        x_ouput = self.datareset(x_mat_centre,x_boundary,select_index)
        return x_ouput,result_centre,result_global
    def getrank(self,rcm):
        e = 0.0000001
        # print(rcm.shape)
        size = rcm.shape[1]
        c = torch.full((size, size), -100000.0)
        c_cuda = c
        if torch.cuda.is_available() :
            # c_cuda = c.to(self.device)
            c_cuda = c.cuda()
        disturb = torch.from_numpy(np.random.normal(0, 0.000001, (size, size)))
        disturb_cuda = disturb.cuda()
        rcm = rcm + disturb_cuda
        maxvalue = torch.max(rcm).detach()
        rcm_flatten = torch.squeeze(rcm.detach().reshape(1,-1),0)
        max2, _ = torch.topk(rcm_flatten, 2)
        sec_maxvalue = max2[1]
        if (maxvalue == sec_maxvalue):
            print("-----------------------")
            print('rcm',rcm)
            print('max',maxvalue)
            print('sec',sec_maxvalue)
            print("-----------------------")
        b = torch.relu(rcm - (maxvalue + sec_maxvalue) /2 ) / ((maxvalue - sec_maxvalue) /2 )
        b_max = torch.max(b).detach()
        b = b/ b_max
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
        assert max_b_value == 1.0 , "maxvalue of  b not 1 is : " + str(max_b_value)
        # if(max_b_value != 1.0):
        #     print(b)
        b_value = b_value.squeeze(0)
        b_value = b_value.to(torch.float32)
        cmul = torch.mm(b_value, c_cuda) + torch.mm(c_cuda, b_value)
        cmul = cmul.unsqueeze(0)
        rcmget = rcm + cmul
        return b , rcmget
    def minmaxscaler(self,rcm):
        max = torch.max(rcm)
        min = torch.min(rcm)

        return (rcm.data-min) /(max-min)
    def fea_extarea(self,batchdata,feature_extractor,batch_seq):
        _, _, _, _ ,fea_map = feature_extractor(batchdata)

        avgout = torch.mean(fea_map, dim=1, keepdim=True)
        # self.tensor2img(avgout[0], 'featuremap' + str(batch_seq))
        avgout = self.avgpool1(avgout)
        avgout = avgout.flatten(1)
        _ ,selectindex = torch.topk(avgout,16,dim=1)
        # 区分不同大小的patch
        datacentral_small_list = []
        datacentral_large_list = []
        databoundary_list = []
        for i in range(batchdata.shape[0]):
            # 区分不同大小的patch
            indices_central_small = selectindex[i,0:4].cuda()
            indices_central_large = selectindex[i, 4:8].cuda()
            centre_list = selectindex[i, 0:8].tolist()
            all_list = [i for i in range(16)]
            boundary_list = [i for i in all_list if i not in centre_list]
            indices_boundary = torch.tensor(boundary_list).cuda()
            # indices_central = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]).cuda()
            # indices_boundary = torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]).cuda()
            batchdata_cur = batchdata[i]
            batchdata_cur = rearrange(batchdata_cur, 'c (h p_h) (w p_w)  -> c  (p_h p_w) (h w) ', p_h=112, p_w=112)
            patchselect_boundary = torch.index_select(batchdata_cur, 2, indices_boundary)
            patchselect_central_small = torch.index_select(batchdata_cur, 2, indices_central_small)
            patchselect_central_large = torch.index_select(batchdata_cur, 2, indices_central_large)
            # 对中心区域先复原在分割
            # 3  * 12544  * 4 -> 3 * 224 * 224
            patchselect_central_small = rearrange(patchselect_central_small, 'c (p_h p_w) (h w) -> c (h p_h) (w p_w)', h=2, w=2,
                                            p_h=112, p_w=112)
            # 3 * 224 * 224 -> 3  * 3136 * 16
            patchselect_central_small = rearrange(patchselect_central_small, 'c (h p_h) (w p_w)-> c (p_h p_w) (h w) ', h=4, w=4)

            datacentral_small_list.append(patchselect_central_small.unsqueeze(0))
            datacentral_large_list.append(patchselect_central_large.unsqueeze(0))
            databoundary_list.append(patchselect_boundary.unsqueeze(0))
        # b * 3 * 12544 * 8
        batchdata_boundary = torch.cat(databoundary_list, dim=0)
        # b * 3 * 3136 * 16
        batchdata_central_small = torch.cat(datacentral_small_list, dim=0)
        # b * 3 * 12544 * 4
        batchdata_central_large = torch.cat(datacentral_large_list, dim=0)
        return batchdata_central_small, batchdata_central_large ,batchdata_boundary ,selectindex
    def datareset(self, batchdata_centre, batchdata_boundary,central_area):
        # batchdata_centre 转化成同一尺寸   batchdata8 b * 3 * 3136 * 16  -> b * 3  *( 112 * 112) * 4
        # # 先复原 在分割
        # # b * 3 * 3136 * 16  -> b * 3  * 16 * 56 * 56
        # batchdata_centre = rearrange(batchdata_centre, 'b c (p_h p_w) l  -> b c l p_h p_w ', p_h=56, p_w=56)
        # # b * 3 * 16 * 56 * 56 -> b * 3 * 224 * 224
        # batchdata_centre = rearrange(batchdata_centre, 'b c (h w) p_h p_w -> b c (h p_h) (w p_w) ', h=4, w=4)
        # # b * 3 * 224 * 224 -> b * 3 * (112 * 112) * 4
        # batchdata_centre = rearrange(batchdata_centre, 'b c (h p_h) (w p_w) -> b c (p_h p_w) (h w)', h=2, w=2)
        batchdata = torch.cat((batchdata_centre, batchdata_boundary), dim=3)
        # index = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7]
        # batchdata = batchdata[:, :, :, index]
        batchsize = batchdata.shape[0]
        # 可以优化
        for i in range(batchsize):
            index = [ -1 for i in range(16)]
            for j in range(8):
                index[central_area[i,j]] = j
            start = 8
            for k in range(16):
                if index[k] == -1:
                    index[k] = start
                    start = start + 1
            # index = indexdict[central_area[i]]
            batchdata[i] = batchdata[i, :, :, index]

        x_out = rearrange(batchdata, 'b c (p_h p_w) (h w) -> b c (h p_h) (w p_w)', h=4, w=4, p_h=112, p_w=112)
        return x_out
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

# 不用 two-step的方式做 mix-scale v9  同时使用特征提取的方式确定中心区域 并且只在中心区域上学习打乱
class  ald_fev9_onlycentre(nn.Module):

    def __init__(self,num_train):
        super(ald_fev9_onlycentre, self).__init__()
        # self.params_boundary = Parameter(torch.Tensor(num_train,12,12), requires_grad=True)
        self.params_centre = Parameter(torch.Tensor(num_train,24,24), requires_grad=True)
        self.avgpool1 = nn.AdaptiveAvgPool2d(4)
        # torch.nn.init.normal_(self.params_boundary)
        torch.nn.init.normal_(self.params_centre)


    def forward(self, x,batch_seq,batchsize,feature_extractor):
        x_centre, x_boundary, select_index = self.fea_extarea(x,feature_extractor,batch_seq)

        if x_centre.shape[0] == batchsize :
            batchnum = batchsize
        else:
            batchnum = x_centre.shape[0]

        start = batch_seq*batchsize
        end   = batch_seq*batchsize + batchnum

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
        for i in range(start,end):
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
            result_r = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 +
                      b11 + b12 + b13 + b14 + b15 + b16 + b17 + b18 + b19 + b20 + b21 + b22 + b23 + b24)
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_centre = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic,0)
                result_centre = torch.cat((result_single_pic,result_centre),0)

        # result_boundary = result_boundary.to(torch.float32)
        result_centre = result_centre.to(torch.float32)

        x_mat_centre = x_centre.matmul(result_centre)
        # x_mat_boundary = x_boundary.matmul(result_boundary)

        x_ouput = self.datareset(x_mat_centre,x_boundary,select_index)
        return x_ouput,result_centre,select_index
    def getrank(self,rcm):
        e = 0.0000001
        # print(rcm.shape)
        size = rcm.shape[1]
        c = torch.full((size, size), -100000.0)
        c_cuda = c
        if torch.cuda.is_available() :
            # c_cuda = c.to(self.device)
            c_cuda = c.cuda()
        disturb = torch.from_numpy(np.random.normal(0, 0.000001, (size, size)))
        disturb_cuda = disturb.cuda()
        rcm = rcm + disturb_cuda
        maxvalue = torch.max(rcm).detach()
        rcm_flatten = torch.squeeze(rcm.detach().reshape(1,-1),0)
        max2, _ = torch.topk(rcm_flatten, 2)
        sec_maxvalue = max2[1]
        if (maxvalue == sec_maxvalue):
            print("-----------------------")
            print('rcm',rcm)
            print('max',maxvalue)
            print('sec',sec_maxvalue)
            print("-----------------------")
        b = torch.relu(rcm - (maxvalue + sec_maxvalue) /2 ) / ((maxvalue - sec_maxvalue) /2 )
        b_max = torch.max(b).detach()
        b = b/ b_max
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
        assert max_b_value == 1.0 , "maxvalue of  b not 1 is : " + str(max_b_value)
        # if(max_b_value != 1.0):
        #     print(b)
        b_value = b_value.squeeze(0)
        b_value = b_value.to(torch.float32)
        cmul = torch.mm(b_value, c_cuda) + torch.mm(c_cuda, b_value)
        cmul = cmul.unsqueeze(0)
        rcmget = rcm + cmul
        return b , rcmget
    def minmaxscaler(self,rcm):
        max = torch.max(rcm)
        min = torch.min(rcm)

        return (rcm.data-min) /(max-min)
    def fea_extarea(self,batchdata,feature_extractor,batch_seq):
        _, _, _, _ ,fea_map = feature_extractor(batchdata)

        avgout = torch.mean(fea_map, dim=1, keepdim=True)
        # self.tensor2img(avgout[0], 'featuremap' + str(batch_seq))
        avgout = self.avgpool1(avgout)
        avgout = avgout.flatten(1)
        _ ,selectindex = torch.topk(avgout,16,dim=1)
        datacentral_list = []
        databoundary_list = []
        for i in range(batchdata.shape[0]):

            indices_central = selectindex[i,0:6].cuda()
            centre_list = selectindex[i, 0:6].tolist()
            all_list = [i for i in range(16)]
            boundary_list = [i for i in all_list if i not in centre_list]
            # random.shuffle(boundary_list)
            indices_boundary = torch.tensor(boundary_list).cuda()
            # indices_central = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]).cuda()
            # indices_boundary = torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]).cuda()
            batchdata_cur = batchdata[i]
            batchdata_cur = rearrange(batchdata_cur, 'c (h p_h) (w p_w)  -> c  (p_h p_w) (h w) ', p_h=112, p_w=112)
            patchselect_boundary = torch.index_select(batchdata_cur, 2, indices_boundary)
            patchselect_central = torch.index_select(batchdata_cur, 2, indices_central)
            # 对中心区域先复原在分割
            # 3  * 12544  * 6 -> 3 * 224 * 336
            patchselect_central = rearrange(patchselect_central, 'c (p_h p_w) (h w) -> c (h p_h) (w p_w)', h=2, w=3,
                                            p_h=112, p_w=112)
            # 3 * 224 * 336 -> 3  * 3136 * 24
            patchselect_central = rearrange(patchselect_central, 'c (h p_h) (w p_w)-> c (p_h p_w) (h w) ', h=4, w=6)

            datacentral_list.append(patchselect_central.unsqueeze(0))
            databoundary_list.append(patchselect_boundary.unsqueeze(0))
        # b * 3 * 12544 * 10
        batchdata_boundary = torch.cat(databoundary_list, dim=0)
        # b * 3 * 3136 * 24
        batchdata_central = torch.cat(datacentral_list, dim=0)
        return batchdata_central, batchdata_boundary ,selectindex
    def datareset(self, batchdata_centre, batchdata_boundary,central_area):
        # batchdata_centre 转化成同一尺寸   batchdata8 b * 3 * 3136 * 24  -> b * 3  *( 112 * 112) * 6
        # 先复原 在分割
        # b * 3 * 3136 * 24  -> b * 3  * 24 * 56 * 56
        batchdata_centre = rearrange(batchdata_centre, 'b c (p_h p_w) l  -> b c l p_h p_w ', p_h=56, p_w=56)
        # b * 3 * 24 * 56 * 56 -> b * 3 * 224 * 336
        batchdata_centre = rearrange(batchdata_centre, 'b c (h w) p_h p_w -> b c (h p_h) (w p_w) ', h=4, w=6)
        # b * 3 * 224 * 336 -> b * 3 * (112 * 112) * 6
        batchdata_centre = rearrange(batchdata_centre, 'b c (h p_h) (w p_w) -> b c (p_h p_w) (h w)', h=2, w=3)
        batchdata = torch.cat((batchdata_centre, batchdata_boundary), dim=3)
        # index = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7]
        # batchdata = batchdata[:, :, :, index]
        batchsize = batchdata.shape[0]
        # 可以优化
        for i in range(batchsize):
            index = [ -1 for i in range(16)]
            for j in range(6):
                index[central_area[i,j]] = j
            start = 6
            for k in range(16):
                if index[k] == -1:
                    index[k] = start
                    start = start + 1
            # index = indexdict[central_area[i]]
            batchdata[i] = batchdata[i, :, :, index]

        x_out = rearrange(batchdata, 'b c (p_h p_w) (h w) -> b c (h p_h) (w p_w)', h=4, w=4, p_h=112, p_w=112)
        return x_out
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

# 做 mix-scale v1  同时使用特征提取的方式确定中心区域 并且只在中心区域上学习打乱
class  ald_fev1_mulsca(nn.Module):

    def __init__(self,num_train):
        super(ald_fev1_mulsca, self).__init__()
        # self.params_boundary = Parameter(torch.Tensor(num_train,12,12), requires_grad=True)
        self.params_centre = Parameter(torch.Tensor(num_train,16,16), requires_grad=True)
        self.params_boundary = Parameter(torch.Tensor(num_train,12,12), requires_grad=True)
        self.avgpool1 = nn.AdaptiveAvgPool2d(4)
        # torch.nn.init.normal_(self.params_boundary)
        torch.nn.init.normal_(self.params_centre)
        torch.nn.init.normal_(self.params_boundary)


    def forward(self, x,batch_seq,batchsize,feature_extractor):
        x_centre, x_boundary, select_index = self.fea_extarea(x,feature_extractor,batch_seq)

        if x_centre.shape[0] == batchsize :
            batchnum = batchsize
        else:
            batchnum = x_centre.shape[0]

        start = batch_seq*batchsize
        end   = batch_seq*batchsize + batchnum

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
        for i in range(start,end):
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
            result_r = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 +
                      b11 + b12 + b13 + b14 + b15 + b16)
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_centre = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic,0)
                result_centre = torch.cat((result_single_pic,result_centre),0)

        first = True
        for i in range(start,end):
            rcm = self.params_boundary[i]
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

            result_r = (b1 + b2 + b3 + b4  + b5 + b6 + b7 + b8 + b9 + b10 +
                      b11 + b12 )
            result_g = result_r
            result_b = result_r
            result_single_pic = torch.cat((result_r, result_g, result_b), dim=0)

            if first:
                result_boundary = torch.unsqueeze(result_single_pic, 0)
                first = False
            else:
                result_single_pic = torch.unsqueeze(result_single_pic,0)
                result_boundary = torch.cat((result_single_pic,result_boundary),0)

        result_boundary = result_boundary.to(torch.float32)
        result_centre = result_centre.to(torch.float32)
        # 分别乘上对应的打乱矩阵
        x_mat_centre = x_centre.matmul(result_centre)
        x_mat_boundary = x_boundary.matmul(result_boundary)
        # 复原图片
        x_ouput = self.datareset(x_mat_centre,x_mat_boundary,select_index)
        return x_ouput,result_centre,select_index
    def getrank(self,rcm):
        e = 0.0000001
        # print(rcm.shape)
        size = rcm.shape[1]
        c = torch.full((size, size), -100000.0)
        c_cuda = c
        if torch.cuda.is_available() :
            # c_cuda = c.to(self.device)
            c_cuda = c.cuda()
        disturb = torch.from_numpy(np.random.normal(0, 0.000001, (size, size)))
        disturb_cuda = disturb.cuda()
        rcm = rcm + disturb_cuda
        maxvalue = torch.max(rcm).detach()
        rcm_flatten = torch.squeeze(rcm.detach().reshape(1,-1),0)
        max2, _ = torch.topk(rcm_flatten, 2)
        sec_maxvalue = max2[1]
        if (maxvalue == sec_maxvalue):
            print("-----------------------")
            print('rcm',rcm)
            print('max',maxvalue)
            print('sec',sec_maxvalue)
            print("-----------------------")
        b = torch.relu(rcm - (maxvalue + sec_maxvalue) /2 ) / ((maxvalue - sec_maxvalue) /2 )
        b_max = torch.max(b).detach()
        b = b/ b_max
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
        assert max_b_value == 1.0 , "maxvalue of  b not 1 is : " + str(max_b_value)
        # if(max_b_value != 1.0):
        #     print(b)
        b_value = b_value.squeeze(0)
        b_value = b_value.to(torch.float32)
        cmul = torch.mm(b_value, c_cuda) + torch.mm(c_cuda, b_value)
        cmul = cmul.unsqueeze(0)
        rcmget = rcm + cmul
        return b , rcmget
    def minmaxscaler(self,rcm):
        max = torch.max(rcm)
        min = torch.min(rcm)

        return (rcm.data-min) /(max-min)
    def fea_extarea(self,batchdata,feature_extractor,batch_seq):
        _, _, _, _ ,fea_map = feature_extractor(batchdata)

        avgout = torch.mean(fea_map, dim=1, keepdim=True)
        # self.tensor2img(avgout[0], 'featuremap' + str(batch_seq))
        avgout = self.avgpool1(avgout)
        avgout = avgout.flatten(1)
        _ ,selectindex = torch.topk(avgout,16,dim=1)
        datacentral_list = []
        databoundary_list = []
        for i in range(batchdata.shape[0]):

            indices_central = selectindex[i,0:4].cuda()
            centre_list = selectindex[i, 0:4].tolist()
            all_list = [i for i in range(16)]
            boundary_list = [i for i in all_list if i not in centre_list]
            indices_boundary = torch.tensor(boundary_list).cuda()
            # indices_central = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]).cuda()
            # indices_boundary = torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]).cuda()
            batchdata_cur = batchdata[i]
            batchdata_cur = rearrange(batchdata_cur, 'c (h p_h) (w p_w)  -> c  (p_h p_w) (h w) ', p_h=112, p_w=112)
            patchselect_boundary = torch.index_select(batchdata_cur, 2, indices_boundary)
            patchselect_central = torch.index_select(batchdata_cur, 2, indices_central)
            # 对中心区域先复原在分割
            # 3  * 12544  * 4 -> 3 * 224 * 224
            patchselect_central = rearrange(patchselect_central, 'c (p_h p_w) (h w) -> c (h p_h) (w p_w)', h=2, w=2,
                                            p_h=112, p_w=112)
            # 3 * 224 * 224 -> 3  * 3136 * 16
            patchselect_central = rearrange(patchselect_central, 'c (h p_h) (w p_w)-> c (p_h p_w) (h w) ', h=4, w=4)

            datacentral_list.append(patchselect_central.unsqueeze(0))
            databoundary_list.append(patchselect_boundary.unsqueeze(0))
        # b * 3 * 12544 * 12
        batchdata_boundary = torch.cat(databoundary_list, dim=0)
        # b * 3 * 3136 * 16
        batchdata_central = torch.cat(datacentral_list, dim=0)
        return batchdata_central, batchdata_boundary ,selectindex
    def datareset(self, batchdata_centre, batchdata_boundary,central_area):
        # batchdata_centre 转化成同一尺寸   batchdata8 b * 3 * 3136 * 16  -> b * 3  *( 112 * 112) * 4
        # 先复原 在分割
        # b * 3 * 3136 * 16  -> b * 3  * 16 * 56 * 56
        batchdata_centre = rearrange(batchdata_centre, 'b c (p_h p_w) l  -> b c l p_h p_w ', p_h=56, p_w=56)
        # b * 3 * 16 * 56 * 56 -> b * 3 * 224 * 224
        batchdata_centre = rearrange(batchdata_centre, 'b c (h w) p_h p_w -> b c (h p_h) (w p_w) ', h=4, w=4)
        # b * 3 * 224 * 224 -> b * 3 * (112 * 112) * 4
        batchdata_centre = rearrange(batchdata_centre, 'b c (h p_h) (w p_w) -> b c (p_h p_w) (h w)', h=2, w=2)
        batchdata = torch.cat((batchdata_centre, batchdata_boundary), dim=3)
        # index = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7]
        # batchdata = batchdata[:, :, :, index]
        batchsize = batchdata.shape[0]
        # 可以优化
        for i in range(batchsize):
            index = [ -1 for i in range(16)]
            for j in range(4):
                index[central_area[i,j]] = j
            start = 4
            for k in range(16):
                if index[k] == -1:
                    index[k] = start
                    start = start + 1
            # index = indexdict[central_area[i]]
            batchdata[i] = batchdata[i, :, :, index]

        x_out = rearrange(batchdata, 'b c (p_h p_w) (h w) -> b c (h p_h) (w p_w)', h=4, w=4, p_h=112, p_w=112)
        return x_out
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


# vgg16 = models.vgg16(pretrained=True)
# # 获取VGG16的特征提取层
# vgg = vgg16.features
# # 将vgg16的特征提取层参数冻结，不对其进行更细腻
# for param in vgg.parameters():
#     param.requires_grad_(False)

        # print(avg_att)
        # avg_att = F.interpolate(avg_att,size=[28,28], mode='nearest')
        # print(avg_att)

# x = torch.rand(2,3,12,12)
# print(x)
# model = active_center()
# y = model(x)
# print(y)
#
# b =torch.tensor( [[[[1., 1., 0., 0.],
#           [1., 1., 0., 0.],
#           [0., 0., 0., 0.],
#           [0., 0., 0., 0.]]],
#
#
#         [[[0., 0., 0., 0.],
#           [0., 0., 0., 0.],
#           [0., 0., 1., 1.],
#           [0., 0., 1., 1.]]]])
# c = torch.tensor( [[[[2., 2., 3., 3.],
#           [2., 2., 3., 3.],
#           [4., 4., 5., 5.],
#           [4., 4., 5., 5.]]],
#
#
#         [[[6., 6., 7., 7.],
#           [6., 6., 7., 7.],
#           [8., 8., 9., 9.],
#           [8., 8., 9., 9.]]]])
#
# print(b.mul(c))







