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

class  ald_pic_n(nn.Module):

    def __init__(self,size,num_train):
        super(ald_pic_n, self).__init__()
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
    def getrank(self,rcm,n=0.25):
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
        # b = torch.relu(rcm - (maxvalue + sec_maxvalue) /2 ) / ((maxvalue - sec_maxvalue) /2 )
        b = torch.relu(rcm - (sec_maxvalue + n*(maxvalue-sec_maxvalue))) /( (1-n) *(maxvalue-sec_maxvalue))
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