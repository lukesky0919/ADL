from __future__ import print_function
import os
from PIL import Image
import time
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from mydataset import  *
from utils import *
import pandas as pd
# 训练一张图中有4*4和8*8两种尺度的图像
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # Data
    print('==> Preparing data..')
    anno_root = './images'
    rawdata_root = './images'
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
    if resume:
        net = torch.load(model_path)
    else:
        net = load_model(model_name='resnet50_pmg', pretrain=True, require_grad=True)
    net = net.cuda()
    # netp = torch.nn.DataParallel(net)

    # GPU
    # device = torch.device("cuda:0,1")
    # net.to(device)
    # cudnn.benchmark = True

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': net.classifier_concat.parameters(), 'lr': 0.002},
        {'params': net.conv_block1.parameters(), 'lr': 0.002},
        {'params': net.classifier1.parameters(), 'lr': 0.002},
        {'params': net.conv_block2.parameters(), 'lr': 0.002},
        {'params': net.classifier2.parameters(), 'lr': 0.002},
        {'params': net.conv_block3.parameters(), 'lr': 0.002},
        {'params': net.classifier3.parameters(), 'lr': 0.002},
        {'params': net.features.parameters(), 'lr': 0.0002}

    ],
        momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    max_com_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
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
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

            # Step 1
            optimizer.zero_grad()
            inputs1 = jigsaw_generator_v2(inputs)
            output_1, _, _, _ = net(inputs1)
            loss1 = CELoss(output_1, targets) * 1
            loss1.backward()
            optimizer.step()

            # Step 2
            optimizer.zero_grad()
            inputs2 = jigsaw_generator(inputs, 4)
            _, output_2, _, _ = net(inputs2)
            loss2 = CELoss(output_2, targets) * 1
            loss2.backward()
            optimizer.step()

            # Step 3
            optimizer.zero_grad()
            inputs3 = jigsaw_generator(inputs, 2)
            _, _, output_3, _ = net(inputs3)
            loss3 = CELoss(output_3, targets) * 1
            loss3.backward()
            optimizer.step()

            # Step 4
            optimizer.zero_grad()
            _, _, _, output_concat = net(inputs)
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


        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |\n' % (
                epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1), train_loss3 / (idx + 1),
                train_loss4 / (idx + 1)))
        torch.cuda.synchronize()
        end_time = time.time()
        print("train one epoch time is :%.3fs" % (end_time-start_time))
        if epoch < 5 or epoch >= 80:
            val_acc, val_acc_com, val_loss = test(net, CELoss, 32)
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                # net.cpu()
                store_name = 'acc' + str(max_val_acc)
                path = './' + exp_dir + '/' +store_name + 'model.pth'
                torch.save(net.state_dict(), path)
                # torch.save(net, './' + store_name + 'model.pth')
                # net.to(device)
            if val_acc_com > max_com_acc:
                max_com_acc = val_acc_com
                # net.cpu()
                store_name = 'comacc' + str(max_com_acc)
                path = './' + exp_dir +  '/' + store_name + 'model.pth'
                torch.save(net.state_dict(), path)

            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write('Iteration %d, test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                epoch, val_acc, val_acc_com, val_loss))
        # else:
        #     net.cpu()
        #     torch.save(net, './' + store_name + '/model.pth')
        #     net.to(device)


train(nb_epoch=300,             # number of epoch
         batch_size=16,         # batch size
         store_name='bird4mulscalev2',     # folder for output
         resume=False,          # resume training from checkpoint
         start_epoch=0,         # the start epoch number when you resume the training
         model_path='')         # the saved model where you want to resume the training
