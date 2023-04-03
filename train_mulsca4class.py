from __future__ import print_function
import os
from PIL import Image
import time
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from mydataset import *
from utils import *
import pandas as pd
from mymodel import *
from matplotlib import pyplot as plt
import argparse
# 在一张图混合有4*4和8*8的尺寸下做


# parameters setting
def parse_args():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--epoch', dest='epoch',
                            default=60, type=int)
    parser.add_argument('--ald_lr', dest='ald_lr',
                            default=1e-6, type=float)
    parser.add_argument('--exp_dir', dest='exp_dir',
                            default='./result/exp1', type=str)
    parser.add_argument('--base_lr', dest='base_lr',
                       default= 8e-6, type=float)
    parser.add_argument('--device_id', dest='device_id',
                        default='0,1', type=str)
    # parser.add_argument('--pre', dest='pre',
    #                     default=True, type=)


    args = parser.parse_args()

    return args


def train(nb_epoch, batch_size, exp_dir, resume=False, start_epoch=0, model_path=None, ald_lr=0.002 , base_lr=0.002):
    # setup output
    exp_dir = exp_dir
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    # print(use_cuda)

    # Data
    print('==> Preparing data..')
    anno_root = './images'
    rawdata_root = './images'
    print('lr setting : ald_lr =%.5f | base_lr = %.5f'%(ald_lr,base_lr))
    train_anno = pd.read_csv(os.path.join(anno_root, 'train.txt'),
                             sep=" ",
                             header=None,
                             names=['ImageName', 'label'])

    train_rcm_set = dataset(rawdata_root=rawdata_root,
                            anno=train_anno,
                            train=True)
    trainloader = torch.utils.data.DataLoader(train_rcm_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=16,
                                              collate_fn=collate_fn4train)
    # drop_last=True)
    # 保存图片名字
    save_imgname(trainloader, exp_dir)

    # transform_train = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.RandomCrop(448, padding=8),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # trainset = torchvision.datasets.ImageFolder(root='./dataset/train', transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    # Model
    # if resume:
    #     net = torch.load(model_path)
    # else:
    #     net = load_model(model_name='resnet50_pmg', pretrain=True, require_grad=True)
    num_classes = 200
    num_train = len(trainloader.dataset)
    net = model_mulscale4class( num_classes)
    net = net.cuda()
    # netp = torch.nn.DataParallel(net)

    # GPU
    # device = torch.device("cuda:0,1")
    # net.to(device)
    # cudnn.benchmark = True
    # print(net)
    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': net.model.classifier_concat.parameters(), 'lr': base_lr},
        {'params': net.model.conv_block1.parameters(), 'lr': base_lr},
        {'params': net.model.classifier1.parameters(), 'lr': base_lr},
        {'params': net.model.conv_block2.parameters(), 'lr': base_lr},
        {'params': net.model.classifier2.parameters(), 'lr': base_lr},
        {'params': net.model.conv_block3.parameters(), 'lr': base_lr},
        {'params': net.model.classifier3.parameters(), 'lr': base_lr},
        {'params': net.model.features.parameters(), 'lr': base_lr*0.1},
        {'params': net.layer1.parameters(), 'lr': ald_lr},
        {'params': net.layer2.parameters(), 'lr': ald_lr},
        {'params': net.layer3.parameters(), 'lr': ald_lr}
    ],
        momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    max_com_acc = 0
    # lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002, 0.002, 0.002, 0.002]

    result_2_save = []
    result_4_save = []
    result_mul8_save = []
    result_mul4_save = []

    trainloss_recorder = recorder(exp_dir)
    for epoch in range(start_epoch, nb_epoch):
        torch.cuda.synchronize()
        start_time = time.time()
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        idx = 0

        adjust_learning_rate(optimizer, epoch,ald_lr,base_lr)
        # 打印lr
        # for i in range(11):
        #     print(optimizer.state_dict()['param_groups'][i]['lr'])
        for batch_idx, data in enumerate(trainloader):

            idx = batch_idx
            # if inputs.shape[0] < batch_size:
            #     continue
            inputs, targets, _ = data
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs = inputs.cuda()
                targets = torch.from_numpy(np.array(targets)).cuda()

            # update learning rate
            # for nlr in range(len(optimizer.param_groups)):
            #     optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])
            tra = True
            # Step 1
            optimizer.zero_grad()
            output_1, _, _, _, resultmul_4  ,resultmul_8 = net(inputs, 8, targets, tra)
            # print(3)
            loss1 = CELoss(output_1, targets) * 1
            loss1.backward()
            optimizer.step()
            # 保存每次变化矩阵的结果
            if (epoch <= 29):
                result_mul8_cpu = resultmul_8[:, 0, :, :].detach().cpu()
                result_mul8_save.append(result_mul8_cpu)
            if (epoch <= 29):
                result_mul4_cpu = resultmul_4[:, 0, :, :].detach().cpu()
                result_mul4_save.append(result_mul4_cpu)

            # Step 2
            optimizer.zero_grad()
            _, output_2, _, _, result_4 = net(inputs, 4, targets, tra)
            loss2 = CELoss(output_2, targets) * 1
            loss2.backward()
            optimizer.step()
            # 保存每次变化矩阵的结果
            if (epoch <= 29):
                result_4_cpu = result_4[:, 0, :, :].detach().cpu()
                result_4_save.append(result_4_cpu)

            # Step 3
            optimizer.zero_grad()
            _, _, output_3, _, result_2 = net(inputs, 2, targets, tra)
            loss3 = CELoss(output_3, targets) * 1
            loss3.backward()
            optimizer.step()
            # 保存每次变化矩阵的结果
            if (epoch <= 29):
                result_2_cpu = result_2[:, 0, :, :].detach().cpu()
                result_2_save.append(result_2_cpu)

            # Step 4
            optimizer.zero_grad()
            _, _, _, output_concat = net(inputs, 1, targets, tra)
            concat_loss = CELoss(output_concat, targets) * 2
            concat_loss.backward()
            optimizer.step()

            #  training log
            _, predicted = torch.max(output_concat.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += concat_loss.item()

            if batch_idx % 50 == 0:
                print(
                    'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                        batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                        train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                        100. * float(correct) / total, correct, total))

        if epoch == 29:
            result_2_npy = np.vstack(result_2_save)
            result_4_npy = np.vstack(result_4_save)
            result_mul8_npy = np.vstack(result_mul8_save)
            result_mul4_npy = np.vstack(result_mul4_save)
            print(result_2_npy.shape)
            print(result_4_npy.shape)
            print(result_mul4_npy.shape)
            print(result_mul8_npy.shape)
            resultmul8 = './' + exp_dir + '/' + 'resultmul8.npy'
            resultmul4 = './' + exp_dir + '/' + 'resultmul4.npy'
            result4_4 = './' + exp_dir + '/' + 'result4.npy'
            result2_2 = './' + exp_dir + '/' + 'result2.npy'
            np.save(result2_2, result_2_npy)
            np.save(result4_4, result_4_npy)
            np.save(resultmul8, result_mul8_npy)
            np.save(resultmul4, result_mul4_npy)

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        trainloss_recorder.addloss(train_loss)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |\n' % (
                    epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1),
                    train_loss3 / (idx + 1),
                    train_loss4 / (idx + 1)))
        torch.cuda.synchronize()
        end_time = time.time()
        print("train one epoch time is :%.3fs" % (end_time - start_time))

        val_acc, val_acc_com, val_loss = test_mulscale4class(net, CELoss, 32)

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            # net.cpu()
            store_name = 'acc' + str(max_val_acc)
            path = './' + exp_dir + '/' + store_name + 'model.pth'
            torch.save(net.state_dict(), path)

        if val_acc_com > max_com_acc:
            max_com_acc = val_acc_com
            # net.cpu()
            store_name = 'comacc' + str(max_com_acc)
            path = './' + exp_dir + '/' + store_name + 'model.pth'
            torch.save(net.state_dict(), path)
        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write('Iteration %d, test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                epoch, val_acc, val_acc_com, val_loss))
        # else:
        #     net.cpu()
        #     torch.save(net, './' + store_name + '/model.pth')
        #     net.to(device)
    # 画训练loss的图
    # 画图
    trainloss_recorder.drawloss("trainloss")



def adjust_learning_rate(optimizer, epoch,ald_lr,base_lr):
    step = epoch % 60
    # lr_decay_step = epoch // 60
    # if lr_decay_step == 0 :
    #     lr_ra = 1
    # else:
    #     lr_ra = pow(0.1,lr_decay_step)
    nb_epoch = 30
    if epoch >= 0 and epoch <= 29:
        optimizer.param_groups[0]['lr'] = 0.0
        optimizer.param_groups[1]['lr'] = 0.0
        optimizer.param_groups[2]['lr'] = 0.0
        optimizer.param_groups[3]['lr'] = 0.0
        optimizer.param_groups[4]['lr'] = 0.0
        optimizer.param_groups[5]['lr'] = 0.0
        optimizer.param_groups[6]['lr'] = 0.0
        optimizer.param_groups[7]['lr'] = 0.0
        optimizer.param_groups[8]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, ald_lr)
        optimizer.param_groups[9]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, ald_lr)
        optimizer.param_groups[10]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, ald_lr)
    else:
        optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(epoch - 30, nb_epoch, base_lr)
        optimizer.param_groups[1]['lr'] = cosine_anneal_schedule(epoch - 30, nb_epoch, base_lr)
        optimizer.param_groups[2]['lr'] = cosine_anneal_schedule(epoch - 30, nb_epoch, base_lr)
        optimizer.param_groups[3]['lr'] = cosine_anneal_schedule(epoch - 30, nb_epoch, base_lr)
        optimizer.param_groups[4]['lr'] = cosine_anneal_schedule(epoch - 30, nb_epoch, base_lr)
        optimizer.param_groups[5]['lr'] = cosine_anneal_schedule(epoch - 30, nb_epoch, base_lr)
        optimizer.param_groups[6]['lr'] = cosine_anneal_schedule(epoch - 30, nb_epoch, base_lr)
        optimizer.param_groups[7]['lr'] = cosine_anneal_schedule(epoch - 30, nb_epoch, base_lr*0.1)
        optimizer.param_groups[8]['lr'] = 0.0
        optimizer.param_groups[9]['lr'] = 0.0
        optimizer.param_groups[10]['lr'] = 0.0


def save_imgname(traindata, exp_dir):
    imgname_list = []
    first = True

    for data in traindata:
        img, label, img_names = data
        imgname = np.expand_dims(np.array(img_names), 1)
        imgname_list.append(imgname)

    np_imgname_list = np.concatenate(imgname_list, 0)
    savepath = './' + exp_dir + '/imglist'
    np.save(savepath, np_imgname_list)


if __name__ == '__main__':

    args = parse_args()
    exp_dir = args.exp_dir
    epoch = args.epoch
    ald_lr = args.ald_lr
    base_lr = args.base_lr
    device_id = args.device_id

    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    train(nb_epoch=epoch,  # number of epoch
          batch_size=16,  # batch size
          exp_dir=exp_dir,  # folder for output
          resume=False,  # resume training from checkpoint
          start_epoch=0,  # the start epoch number when you resume the training
          model_path='', # the saved model where you want to resume the training
          ald_lr = ald_lr, # Adaptive learning disruption lr
          base_lr = base_lr # base lr
          )
