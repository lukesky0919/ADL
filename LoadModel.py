import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
from activefunc import activercm ,getpic
from rcmdataprocess import dataprocess ,datareset
from torch.nn.parameter import Parameter
from selflayer import rcmlayer
from DclModel import DclModel
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MainModel(nn.Module):
    def __init__(self,num_classes,num_train,config):
        super(MainModel, self).__init__()
        self.num_classes = num_classes
        self.model = DclModel(num_classes)
        pre = torch.load('DCL180.pth')
        self.model.load_state_dict(pre)
        self.layer1 = rcmlayer(49, num_train,config.outdir,config.device_id)
        # self.model = models.resnet50(pretrained=False)
        # pre = torch.load('resnet50-19c8e357.pth')
        # self.model.load_state_dict(pre)
        # self.model = nn.Sequential(*list(self.model.children())[:-2])
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        # self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        # self.layer1 = rcmlayer(49,num_train)
        # self.classifier_swap = nn.Linear(2048, 2, bias=False)
        # self.Convmask = nn.Conv2d(2048, 1, 1, stride=1, padding=0, bias=True)
        # self.avgpool2 = nn.AvgPool2d(2, stride=2)
    def forward(self,x,use_rcm,batch_seq,batchsize,epoch):


        if use_rcm ==True :

            x_rcm = x
            x_rcm = dataprocess(x_rcm, [7, 7])
            x_rcm ,result = self.layer1(x_rcm,batch_seq,batchsize,epoch)
            x_rcm = datareset(x_rcm, 448, 448, [7, 7])

            x = torch.cat((x,x_rcm),dim=0)

        else :
            result  = 0
        out = self.model(x)
        # mask = self.Convmask(x)
        # mask = self.avgpool2(mask)
        # mask = torch.tanh(mask)
        # mask = mask.view(mask.size(0), -1)
        #
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # out = []
        # #print(x.shape)
        # # 用来计算普通的分类loss
        # out.append(self.classifier(x))
        # # 计算是否是swap过的图片的一个loss
        # out.append(self.classifier_swap(x))
        # # 计算 一个swap后位置之间的loss
        # out.append(mask)
        # #print(x.shape)
        return out ,result



# if __name__ == '__main__':
#     model = MainModel(2)
#     model = model.to(device)
#     for i in range(2):
#         input = torch.randn([2,3,4,4])
#         input = input.to(device)
#         output = model(input)
#         with torch.no_grad():
#             for name, param in model.named_parameters():
#                 if 'rcm.weight' in name:
#                     print(param)
#                     param.copy_(activercm(param.detach()))
#         print(output)
