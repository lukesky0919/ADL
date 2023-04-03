import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
from model import PMG,PMG_fusion,PMG_resnet18,PMG_vgg16
from Resnet import *
from Vgg import *
from mydataset import  *
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from  rcmdataprocess import *
from pmg_channelatt import *
from fusion_feature_extractor import *
from collections import OrderedDict
# from convnext import *
# from PMG_convnext import *
import scipy.io as scio
# import ml_collections
# from transformer_feature_extractor import Transformer
from resnet_withtrans import resnet50_mhsa
def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def load_model(model_name, pretrain=True, require_grad=True,num_class=200):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, num_class)
    if model_name == 'resnet18_pmg':
        net = resnet18(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG_resnet18(net, 512, num_class)
    if model_name == 'vgg16_pmg':
        net = vgg16_bn(pretrained=False)
        self_model_dict = net.state_dict()
        model_path = 'vgg16_bn-6c64b313.pth'
        load_dict = torch.load(model_path)
        key = list(self_model_dict.keys())
        name = list(load_dict.keys())
        weights = list(load_dict.values())
        t = 0
        for i in range(len(weights)):
            # 不加载最后的全连接层
            if 'classifier' in name[i]:
                break
            # 当前模型使用BN层多一个num_batches_tracked，但是加载的模型中没有，因此需要跳过
            if 'num_batches_tracked' in key[i + t]:
                t += 1
            self_model_dict[key[i + t]] = weights[i]

        net.load_state_dict(self_model_dict, strict=True)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG_vgg16(net, 512, num_class)
    # if model_name == 'convnext_pmg':
    #     net = convnext_base(pretrained=pretrain)
    #     for param in net.parameters():
    #         param.requires_grad = require_grad
    #     net = PMG_convnext(net, 512, num_class)
    return net

def load_model_dcl(model_name, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'pmg_dcl':
        net = resnet50(pretrained=True)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG_dcl(net, 512, 200)
        if pretrain == True :
            pre = torch.load('dclacc88.62.pth')
            net.load_state_dict(pre)

    return net

def load_model_rcm(model_name, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 200)
        pre = torch.load('acc88.608.pth')
        net.load_state_dict(pre)
    return net

def load_model_ald(model_name, dset):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=False)
        for param in net.parameters():
            param.requires_grad = True
        if dset == 'cub' :
            net = PMG(net, 512, 200)
            pre = torch.load('comacc89.4.pth')
        if dset == 'air' :
            net = PMG(net, 512, 100)
            pre = torch.load('airacc93.5.pth')
        if dset == 'car' :
            net = PMG(net, 512, 196)
            pre = torch.load('caracc95.1.pth')
        if dset == 'dog' :
            net = PMG(net, 512, 120)
            pre = torch.load('dogacc86.1.pth')

        net_dict = net.state_dict()
        state_dict = {k: v for k, v in pre.items() if k in net_dict.keys()}
        net_dict.update(state_dict)
        net.load_state_dict(net_dict)
    if model_name == 'resnet50_msha_pmg':
        net = resnet50_mhsa(pretrained=False)
        for param in net.parameters():
            param.requires_grad = True
        if dset == 'cub' :
            net = PMG(net, 512, 200)
            pre = torch.load('comacc89.4.pth')
        if dset == 'air' :
            net = PMG(net, 512, 100)
            pre = torch.load('airacc93.5.pth')
        if dset == 'car' :
            net = PMG(net, 512, 196)
            pre = torch.load('caracc95.1.pth')
        if dset == 'dog' :
            net = PMG(net, 512, 120)
            pre = torch.load('dogacc86.1.pth')

        net_dict = net.state_dict()
        state_dict = {k: v for k, v in pre.items() if k in net_dict.keys()}
        net_dict.update(state_dict)
        net.load_state_dict(net_dict)

    if model_name == 'resnet18_pmg':
        net = resnet18(pretrained=False)
        for param in net.parameters():
            param.requires_grad = True
        if dset == 'cub' :
            net = PMG_resnet18(net, 512, 200)
            pre = torch.load('cubacc87.6.pth')
        if dset == 'air' :
            net = PMG_resnet18(net, 512, 100)
            pre = torch.load('resnet18airacc91.5.pth')
        if dset == 'car' :
            net = PMG_resnet18(net, 512, 196)
            pre = torch.load('caracc95.1.pth')
        if dset == 'dog' :
            net = PMG_resnet18(net, 512, 120)
            pre = torch.load('dogacc86.1.pth')

        net_dict = net.state_dict()
        state_dict = {k: v for k, v in pre.items() if k in net_dict.keys()}
        net_dict.update(state_dict)
        net.load_state_dict(net_dict)
    if model_name == 'vgg16_pmg':
        net = vgg16_bn(pretrained=False)
        for param in net.parameters():
            param.requires_grad = True
        if dset == 'cub' :
            net = PMG_vgg16(net, 512, 200)
            pre = torch.load('vgg16cubacc88.8.pth')
        if dset == 'air' :
            net = PMG_vgg16(net, 512, 100)
            pre = torch.load('resnet18airacc91.5.pth')
        if dset == 'car' :
            net = PMG_vgg16(net, 512, 196)
            pre = torch.load('caracc95.1.pth')
        if dset == 'dog' :
            net = PMG_vgg16(net, 512, 120)
            pre = torch.load('dogacc86.1.pth')

        net_dict = net.state_dict()
        state_dict = {k: v for k, v in pre.items() if k in net_dict.keys()}
        net_dict.update(state_dict)
        net.load_state_dict(net_dict)
    if model_name == 'fusion_model':
        #  use both transformer and cnn to extrct feature
        net = resnet50(pretrained=False)
        for param in net.parameters():
            param.requires_grad = True
        if dset == 'cub':
            net = PMG_fusion(net, 512, 200)
            pre = torch.load('comacc89.4.pth')
        if dset == 'air':
            net = PMG_fusion(net, 512, 100)
            pre = torch.load('airacc93.5.pth')
        if dset == 'car':
            net = PMG_fusion(net, 512, 196)
            pre = torch.load('caracc95.1.pth')
        if dset == 'dog':
            net = PMG_fusion(net, 512, 120)
            pre = torch.load('dogacc86.1.pth')

        net_dict = net.state_dict()
        state_dict = {k: v for k, v in pre.items() if k in net_dict.keys()}
        net_dict.update(state_dict)
        net.load_state_dict(net_dict)

    return net

def load_model_aldv2(model_name, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 200)
        path = './bird4mulscalev2/acc88.86.pth'
        pre = torch.load(path)
        net.load_state_dict(pre)
    return net






def load_model_channelatt(model_name, dset):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=False)
        for param in net.parameters():
            param.requires_grad = True
        if dset == 'cub' :
            net = PMG_channelatt(net, 512, 200)
            pre = torch.load('comacc89.4.pth')
        if dset == 'air' :
            net = PMG_channelatt(net, 512, 100)
            pre = torch.load('airacc93.5.pth')
        if dset == 'car' :
            net = PMG_channelatt(net, 512, 196)
            pre = torch.load('caracc95.1.pth')
        net_dict = net.state_dict()
        state_dict = {k: v for k, v in pre.items() if k in net_dict.keys()}
        net_dict.update(state_dict)
        net.load_state_dict(net_dict)

    return net

def load_fusion_features():
    feature_extractor = Fusion_Feature_Extractor(patch_size=32, channel_ratio=4, embed_dim=384, depth=16,
                             num_heads=6, mlp_ratio=4, qkv_bias=True)
    for param in feature_extractor.parameters():
        param.requires_grad = True
    new_state_dict = OrderedDict()
    pretrained_dict = torch.load("./resnet50-19c8e357.pth")
    for i, kv in enumerate(pretrained_dict.items()):
        # print(i,kv[0])
        k = kv[0]
        v = kv[1]
        # if k == 'layer1.0.conv1.weight':
        #     print(v)
        if (i >= 0 and i <= 4):
            name = k
            new_state_dict[name] = v
        if (i >= 5 and i <= 24):
            name = 'conv_1.' + k[9:]
            new_state_dict[name] = v
        elif (i >= 25 and i <= 39):
            name = k[:6] + '.0.cnn_block.' + k[9:]
            new_state_dict[name] = v
        elif (i >= 40 and i <= 54):
            name = k[:6] + '.1.cnn_block.' + k[9:]
            new_state_dict[name] = v
        elif (i >= 55 and i <= 264):
            name = k[:9] + 'cnn_block.' + k[9:]
            new_state_dict[name] = v
    model_dict = feature_extractor.state_dict()
    model_dict.update(new_state_dict)
    # for i, kv in enumerate(new_state_dict.items()):
    #     print(kv[0])
    # for i, kv in enumerate(model_dict.items()):
    #     print(kv[0])
    feature_extractor.load_state_dict(model_dict)

    return feature_extractor
def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws
def jigsaw_generator_mulscale(images):
    # 生成 4*4 和 8*8 两种粒度随机混合的图片
    images_4 ,images_8 = datamulscale4_8(images)
    list4_4 = [images_4[:,:,:,i] for i in range(12)]
    list8_8 = [images_8[:, :, :, i] for i in range(16)]
    random.shuffle(list4_4)
    random.shuffle(list8_8)
    images_4 = torch.stack(list4_4,dim=-1)
    images_8 = torch.stack(list8_8,dim=-1)
    images = datareset4_8(images_4, images_8)
    return images


def jigsaw_generator_v2(images):
    # 生成 v2形式的两种 尺寸随机混合的图片
    images_boundary ,images_centre = datamulscalev2(images)
    list_boundary = [images_boundary[:,:,:,i] for i in range(24)]
    list_centre = [images_centre[:, :, :, i] for i in range(16)]
    random.shuffle(list_boundary)
    random.shuffle(list_centre)
    images_boundary = torch.stack(list_boundary,dim=-1)
    images_centre = torch.stack(list_centre,dim=-1)
    images = dataresetv2(images_boundary, images_centre)
    return images

def jigsaw_generator_v4(images):
    # 生成 v2形式的两种 尺寸随机混合的图片
    images_boundary1, images_boundary2,images_centre = datamulscalev4(images)
    list_boundary1 = [images_boundary1[:,:,:,i] for i in range(4)]
    list_boundary2 = [images_boundary2[:, :, :, i] for i in range(6)]
    list_centre = [images_centre[:, :, :, i] for i in range(16)]
    random.shuffle(list_boundary1)
    random.shuffle(list_boundary2)
    random.shuffle(list_centre)
    images_boundary1 = torch.stack(list_boundary1,dim=-1)
    images_boundary2 = torch.stack(list_boundary2, dim=-1)
    images_centre = torch.stack(list_centre,dim=-1)
    images = dataresetv4(images_boundary1,images_boundary2 ,images_centre)
    return images

def jigsaw_generator_dcl(images, n):
    l = []
    i = 0
    for a in range(n):
        for b in range(n):
            l.append([a, b, i])
            i = i + 1
    block_size = 448 // n
    rounds = n ** 2
    swap_range = n ** 2
    random.shuffle(l)

    # print(l)
    law = [i for i in range(rounds)]
    jigsaws = images.clone()
    for i in range(rounds):
        x, y ,order = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

        tmp = law[0]
        law[0] = law[order]
        law[order] = tmp
    swaplaw = [((index - (swap_range // 2)) / swap_range) for index in law]

    return jigsaws ,swaplaw



# def jigsaw_generator_dcl(images, n):
#     l = []
#     i = 0
#     for a in range(n):
#         for b in range(n):
#             ++i
#             l.append(zip([a, b],i))
#     block_size = 448 // n
#     rounds = n ** 2
#     random.shuffle(l)
#     jigsaws = 0
#     print(l)
#     # jigsaws = images.clone()
#     # for i in range(rounds):
#     #     x, y = l[i]
#     #     temp = jigsaws[..., 0:block_size, 0:block_size].clone()
#     #     jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
#     #                                             y * block_size:(y + 1) * block_size].clone()
#     #     jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp
#
#     return jigsaws
# jigsaw_generator_dcl(8, 2)

def test_ald(net, criterion, batch_size,dset):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    # device = torch.device("cuda:0,1")
    if dset == 'cub':
        anno_root = './images'
        rawdata_root = './images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])
    if dset == 'air':
        anno_root = './data/aircraft'
        rawdata_root = './data/aircraft/images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])
    if dset == 'car':
        anno_root = './data/car'
        rawdata_root = './data/car/images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])
    if dset == 'dog':

        rawdata_root = './data/dog/images'
        data_path = './data/dog/test_list.mat'
        data = scio.loadmat(data_path)
        name = data['file_list']
        label = data['labels'] - 1
        label = label.astype(np.int16)
        split = [item[0][0] for item in name]
        name = np.array(split)
        name = np.expand_dims(name, axis=1)
        save = np.concatenate((name, label), axis=1)
        test_anno = pd.DataFrame(save,columns=['ImageName', 'label'])


    test_set = dataset(rawdata_root=rawdata_root,
                            anno=test_anno,
                            test=True)
    testloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=16,
                                              collate_fn=collate_fn4test)

    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            idx = batch_idx
            inputs, targets, _ = data
            if use_cuda:
                inputs = inputs.cuda()
                targets_int = [int(i) for i in targets]
                targets = torch.from_numpy(np.array(targets_int)).cuda()
            tra = False
            output_1, output_2, output_3, output_concat= net(inputs,'original',batch_idx,batch_size,tra)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss

def test_ald_trans(net, criterion, batch_size,dset):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    # device = torch.device("cuda:0,1")
    if dset == 'cub':
        anno_root = './images'
        rawdata_root = './images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])
    if dset == 'air':
        anno_root = './data/aircraft'
        rawdata_root = './data/aircraft/images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])
    if dset == 'car':
        anno_root = './data/car'
        rawdata_root = './data/car/images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])
    if dset == 'dog':

        rawdata_root = './data/dog/images'
        data_path = './data/dog/test_list.mat'
        data = scio.loadmat(data_path)
        name = data['file_list']
        label = data['labels'] - 1
        label = label.astype(np.int16)
        split = [item[0][0] for item in name]
        name = np.array(split)
        name = np.expand_dims(name, axis=1)
        save = np.concatenate((name, label), axis=1)
        test_anno = pd.DataFrame(save,columns=['ImageName', 'label'])


    test_set = dataset(rawdata_root=rawdata_root,
                            anno=test_anno,
                            test=True)
    testloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=16,
                                              collate_fn=collate_fn4test)

    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)
    # prepare test feature
    trans_pic = np.load('trans_test_imglist.npy')
    trans_pic_list = []
    for i in range(trans_pic.shape[0]):
        trans_pic_list.append(trans_pic[i][0])
    trans_feature_all = load_test_trans_feature()
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            idx = batch_idx
            inputs, targets, label_name = data
            if use_cuda:
                inputs = inputs.cuda()
                targets_int = [int(i) for i in targets]
                targets = torch.from_numpy(np.array(targets_int)).cuda()
            tra = False
            select_index = [trans_pic_list.index(i) for i in label_name]
            indices_select = torch.tensor(select_index)
            x_trans_feature = torch.index_select(trans_feature_all, 0, indices_select)
            x_trans_feature = x_trans_feature.cuda()
            output_1, output_2, output_3, output_concat= net(inputs,'original',batch_idx,batch_size,tra,x_trans_feature)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss

def load_test_trans_feature():
    feature =  np.load('trans_test_feature.npy')
    feature_tensor = torch.from_numpy(feature)
    return feature_tensor

def test(net, criterion, batch_size,data):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    # device = torch.device("cuda:0,1")
    if data == 'cub':
        anno_root = './images'
        rawdata_root = './images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                 sep=" ",
                                 header=None,
                                 names=['ImageName', 'label'])
    if data == 'air':
        anno_root = './data/aircraft'
        rawdata_root = './data/aircraft/images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                 sep=" ",
                                 header=None,
                                 names=['ImageName', 'label'])
    if data == 'car':
        anno_root = './data/car'
        rawdata_root = './data/car/images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                 sep=" ",
                                 header=None,
                                 names=['ImageName', 'label'])
    if data == 'dog':

        rawdata_root = './data/dog/images'
        data_path = './data/dog/test_list.mat'
        data = scio.loadmat(data_path)
        name = data['file_list']
        label = data['labels'] - 1
        label = label.astype(np.int16)
        split = [item[0][0] for item in name]
        name = np.array(split)
        name = np.expand_dims(name, axis=1)
        save = np.concatenate((name, label), axis=1)
        test_anno = pd.DataFrame(save,columns=['ImageName', 'label'])

    test_set = dataset(rawdata_root=rawdata_root,
                            anno=test_anno,
                            test=True)
    testloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=16,
                                              collate_fn=collate_fn4test)

    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            idx = batch_idx
            inputs, targets, _ = data
            if use_cuda:
                inputs = inputs.cuda()
                targets_int = [int(i) for i in targets]
                targets = torch.from_numpy(np.array(targets_int)).cuda()
            tra = False
            output_1, output_2, output_3, output_concat= net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss

def testdcl(net, criterion, batch_size):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    # device = torch.device("cuda:0,1")
    anno_root = './images'
    rawdata_root = './images'
    test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                             sep=" ",
                             header=None,
                             names=['ImageName', 'label'])

    test_set = dataset(rawdata_root=rawdata_root,
                            anno=test_anno,
                            test=True)
    testloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=8,
                                              collate_fn=collate_fn4test)

    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            idx = batch_idx
            inputs, targets, _ = data
            if use_cuda:
                inputs = inputs.cuda()
                targets = torch.from_numpy(np.array(targets)).cuda()
            tra = False
            output_1, output_2, output_3, output_concat, _, _, _, _, _, _= net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            _, predicted_8 = torch.max(output_1.data, 1)
            _, predicted_4 = torch.max(output_2.data, 1)
            _, predicted_2 = torch.max(output_3.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()
            correct_8  = predicted_8.eq(targets.data).cpu().sum()
            correct_4 = predicted_4.eq(targets.data).cpu().sum()
            correct_2 = predicted_2.eq(targets.data).cpu().sum()
            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_acc_8 = 100. * float(correct_8) / total
    test_acc_4 = 100. * float(correct_4) / total
    test_acc_2 = 100. * float(correct_2) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss , test_acc_8 , test_acc_4 ,test_acc_2

def test_8(net, criterion, batch_size):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    # device = torch.device("cuda:0,1")
    anno_root = './images'
    rawdata_root = './images'
    test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                             sep=" ",
                             header=None,
                             names=['ImageName', 'label'])

    test_set = dataset(rawdata_root=rawdata_root,
                            anno=test_anno,
                            test=True)
    testloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=8,
                                              collate_fn=collate_fn4test)

    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            idx = batch_idx
            inputs, targets, _ = data
            if use_cuda:
                inputs = inputs.cuda()
                targets = torch.from_numpy(np.array(targets)).cuda()
            tra = False
            output_1, output_2, output_3, output_concat ,_,_,_,_,_,_= net(inputs,1,batch_idx,batch_size,tra)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss

class recorder:
    def __init__(self,dir):
        self.lossrecoder = []
        self.dir = dir
    def addloss(self,loss):
        self.lossrecoder.append(loss)
    def drawloss(self,name):
        # 画训练loss的图
        # 画图

        x = np.arange(1, len(self.lossrecoder) + 1)
        y = np.array(self.lossrecoder)
        plt.title(name)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(x, y)
        plt.savefig('./' + self.dir + '/' + name +'.jpg')
        plt.close()

def test4class(net, criterion, batch_size,dset):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    # device = torch.device("cuda:0,1")
    if dset == 'cub':
        anno_root = './images'
        rawdata_root = './images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])
    if dset == 'air':
        anno_root = './data/aircraft'
        rawdata_root = './data/aircraft/images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])
    if dset == 'car':
        anno_root = './data/car'
        rawdata_root = './data/car/images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])

    test_set = dataset(rawdata_root=rawdata_root,
                            anno=test_anno,
                            test=True)
    testloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=16,
                                              collate_fn=collate_fn4test)

    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            idx = batch_idx
            inputs, targets, _ = data
            if use_cuda:
                inputs = inputs.cuda()
                targets = torch.from_numpy(np.array(targets)).cuda()
            tra = False
            output_1, output_2, output_3, output_concat= net(inputs,'original',targets,tra)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss

def test_mul48(net, criterion, batch_size):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    # device = torch.device("cuda:0,1")
    anno_root = './images'
    rawdata_root = './images'
    test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                             sep=" ",
                             header=None,
                             names=['ImageName', 'label'])

    test_set = dataset(rawdata_root=rawdata_root,
                            anno=test_anno,
                            test=True)
    testloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=16,
                                              collate_fn=collate_fn4test)

    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            idx = batch_idx
            inputs, targets, _ = data
            if use_cuda:
                inputs = inputs.cuda()
                targets = torch.from_numpy(np.array(targets)).cuda()
            tra = False
            output_1, output_2, output_3, output_concat= net(inputs,'1',batch_idx,batch_size,tra)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss

def test_mulscale4class(net, criterion, batch_size):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    # device = torch.device("cuda:0,1")
    anno_root = './images'
    rawdata_root = './images'
    test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                             sep=" ",
                             header=None,
                             names=['ImageName', 'label'])

    test_set = dataset(rawdata_root=rawdata_root,
                            anno=test_anno,
                            test=True)
    testloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=16,
                                              collate_fn=collate_fn4test)

    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            idx = batch_idx
            inputs, targets, _ = data
            if use_cuda:
                inputs = inputs.cuda()
                targets = torch.from_numpy(np.array(targets)).cuda()
            tra = False
            output_1, output_2, output_3, output_concat= net(inputs,1,targets,tra)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss

def test_mulv2(net, criterion, batch_size):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    # device = torch.device("cuda:0,1")
    anno_root = './images'
    rawdata_root = './images'
    test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                             sep=" ",
                             header=None,
                             names=['ImageName', 'label'])

    test_set = dataset(rawdata_root=rawdata_root,
                            anno=test_anno,
                            test=True)
    testloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=16,
                                              collate_fn=collate_fn4test)

    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            idx = batch_idx
            inputs, targets, _ = data
            if use_cuda:
                inputs = inputs.cuda()
                targets = torch.from_numpy(np.array(targets)).cuda()
            tra = False
            output_1, output_2, output_3, output_concat= net(inputs,'1',batch_idx,batch_size,tra)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss

def test_mulv4(net, criterion, batch_size):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    # device = torch.device("cuda:0,1")
    anno_root = './images'
    rawdata_root = './images'
    test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                             sep=" ",
                             header=None,
                             names=['ImageName', 'label'])

    test_set = dataset(rawdata_root=rawdata_root,
                            anno=test_anno,
                            test=True)
    testloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=16,
                                              collate_fn=collate_fn4test)

    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            idx = batch_idx
            inputs, targets, _ = data
            if use_cuda:
                inputs = inputs.cuda()
                targets = torch.from_numpy(np.array(targets)).cuda()
            tra = False
            output_1, output_2, output_3, output_concat= net(inputs,'1',batch_idx,batch_size,tra)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss

def test4dataset(net, criterion, batch_size):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    # device = torch.device("cuda:0,1")
    anno_root = './images'
    rawdata_root = './images'
    test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                             sep=" ",
                             header=None,
                             names=['ImageName', 'label'])

    test_set = dataset(rawdata_root=rawdata_root,
                            anno=test_anno,
                            test=True)
    testloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=16,
                                              collate_fn=collate_fn4test)

    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            idx = batch_idx
            inputs, targets, _ = data
            if use_cuda:
                inputs = inputs.cuda()
                targets = torch.from_numpy(np.array(targets)).cuda()
            tra = False
            output_1, output_2, output_3, output_concat= net(inputs, 1, tra)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss


def test_mulv1(net, criterion, batch_size,dset):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    # device = torch.device("cuda:0,1")
    if dset == 'cub':
        anno_root = './images'
        rawdata_root = './images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])
    if dset == 'air':
        anno_root = './data/aircraft'
        rawdata_root = './data/aircraft/images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])
    if dset == 'car':
        anno_root = './data/car'
        rawdata_root = './data/car/images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])

    test_set = dataset(rawdata_root=rawdata_root,
                            anno=test_anno,
                            test=True)
    testloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=16,
                                              collate_fn=collate_fn4test)

    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            idx = batch_idx
            inputs, targets, _ = data
            if use_cuda:
                inputs = inputs.cuda()
                targets = torch.from_numpy(np.array(targets)).cuda()
            tra = False
            output_1, output_2, output_3, output_concat= net(inputs,'original',batch_idx,batch_size,tra)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss

def test_mixscav1_random(net, criterion, batch_size,dset):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    # device = torch.device("cuda:0,1")
    if dset == 'cub':
        anno_root = './images'
        rawdata_root = './images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])
    if dset == 'air':
        anno_root = './data/aircraft'
        rawdata_root = './data/aircraft/images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])
    if dset == 'car':
        anno_root = './data/car'
        rawdata_root = './data/car/images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])


    test_set = dataset(rawdata_root=rawdata_root,
                            anno=test_anno,
                            test=True)
    testloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=16,
                                              collate_fn=collate_fn4test)

    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            idx = batch_idx
            inputs, targets, _ = data
            if use_cuda:
                inputs = inputs.cuda()
                targets = torch.from_numpy(np.array(targets)).cuda()
            tra = False
            output_1, output_2, output_3, output_concat= net(inputs,1,batch_idx,batch_size,tra)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss

def test_chanatt(net, criterion, batch_size,dset):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    # device = torch.device("cuda:0,1")
    if dset == 'cub':
        anno_root = './images'
        rawdata_root = './images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])
    if dset == 'air':
        anno_root = './data/aircraft'
        rawdata_root = './data/aircraft/images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])
    if dset == 'car':
        anno_root = './data/car'
        rawdata_root = './data/car/images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])


    test_set = dataset(rawdata_root=rawdata_root,
                            anno=test_anno,
                            test=True)
    testloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=16,
                                              collate_fn=collate_fn4test)

    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            idx = batch_idx
            inputs, targets, _ = data
            if use_cuda:
                inputs = inputs.cuda()
                targets = torch.from_numpy(np.array(targets)).cuda()
            tra = False
            output_1, output_2, output_3, output_concat= net(inputs,'original',batch_idx,batch_size,tra)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss

def test_fusmodel(net, criterion, batch_size,dset):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    # device = torch.device("cuda:0,1")
    if dset == 'cub':
        anno_root = './images'
        rawdata_root = './images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])
    if dset == 'air':
        anno_root = './data/aircraft'
        rawdata_root = './data/aircraft/images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])
    if dset == 'car':
        anno_root = './data/car'
        rawdata_root = './data/car/images'
        test_anno = pd.read_csv(os.path.join(anno_root, 'test.txt'),
                                sep=" ",
                                header=None,
                                names=['ImageName', 'label'])


    test_set = dataset(rawdata_root=rawdata_root,
                            anno=test_anno,
                            test=True)
    testloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16,
                                              collate_fn=collate_fn4test)

    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # testset = torchvision.datasets.ImageFolder(root='./dataset/test',
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            idx = batch_idx
            inputs, targets, _ = data
            if use_cuda:
                inputs = inputs.cuda()
                targets = torch.from_numpy(np.array(targets)).cuda()
            tra = False
            output_1, output_2, output_3, output_concat , output_trans = net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat + output_trans

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted_trans = torch.max(output_trans.data, 1)
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()
            correct_trans = predicted_trans.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d) | Trans Acc : %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total,
                100. * float(correct_trans) / total, correct_trans,total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_acc_tran = 100. * float(correct_trans) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss , test_acc_tran

# feature_extractor = load_fusion_features()