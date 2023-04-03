import os
import torch
import torch.utils.data as data
import pandas
import random
import PIL.Image as Image
from PIL import ImageStat
# from transforms import transforms
from torchvision import transforms, models
import numpy as np
class dataset(data.Dataset):
    def __init__(self, rawdata_root,anno , train=False,
                 train_val=False, test=False,numcls=200):
        self.root_path = rawdata_root
        if isinstance(anno, pandas.core.frame.DataFrame):
            self.paths = anno['ImageName'].tolist()
            self.labels = anno['label'].tolist()
        elif isinstance(anno, dict):
            self.paths = anno['img_name']
            self.labels = anno['label']
        self.train = train
        self.test = test
        self.use_cls_2 = False
        self.use_cls_mul = True
        self.numcls = numcls
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])
        img = self.pil_loader(img_path)
        label = self.labels[item]
        label = int(label)
        if self.test:
            transform_test = transforms.Compose([
                transforms.Resize((550, 550)),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            img = transform_test(img)
            return img, label, self.paths[item]

        # 对图像进行通常的数据增强
        transform_train = transforms.Compose([
            transforms.Resize((550, 550)),
            transforms.RandomCrop(448, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        img_unswap = transform_train(img)
        # print(img_unswap.shape)
        # img_swap_8, swaplaw_8 = self.jigsaw_generator_dcl(img_unswap, 8)
        # return img_unswap, label, self.paths[item]
        swap_range = 4
        unswap_law_2 = [(i - (swap_range // 2)) / swap_range for i in range(swap_range)]
        swap_range = 16
        unswap_law_4 = [(i - (swap_range // 2)) / swap_range for i in range(swap_range)]
        swap_range = 64
        unswap_law_8 = [(i - (swap_range // 2)) / swap_range for i in range(swap_range)]
        # unswap_law = [unswap_law_2,unswap_law_8,unswap_law_8]
        img_swap_8, swaplaw_8   =  self.jigsaw_generator_dcl(img_unswap,8)
        img_swap_4, swaplaw_4   = self.jigsaw_generator_dcl(img_unswap, 4)
        img_swap_2, swaplaw_2   = self.jigsaw_generator_dcl(img_unswap, 2)
        label = self.labels[item]
        # swap_law = [swaplaw_2,swaplaw_4,swaplaw_8]
        # img_swap = [img_swap_2,img_swap_4,img_swap_8]

        return img_unswap, img_swap_2, img_swap_4, img_swap_8, unswap_law_2, unswap_law_4, unswap_law_8 ,swaplaw_2, swaplaw_4,swaplaw_8 ,label,  self.paths[item]

        # # 将图片按照 8x8 4x4 2x2 裁剪
        # swap_size =
        # image_unswap_list = self.crop_image(img_unswap, self.swap_size)
        #
        # swap_range = self.swap_size[0] * self.swap_size[1]
        # swap_law1 = [(i - (swap_range // 2)) / swap_range for i in range(swap_range)]
        # if self.train:
        #
        #     img_swap  = self.swap(img_unswap)
        #     img_unswap_np = img_unswap
        #     swap_law3 = self.get_index(img_unswap_np)
        #     image_swap_list = self.crop_image(img_swap, self.swap_size)
        #     unswap_stats = [sum(ImageStat.Stat(im).mean) for im in image_unswap_list]
        #     swap_stats = [sum(ImageStat.Stat(im).mean) for im in image_swap_list]
        #     swap_law2 = []
        #     # swap_law3 = []
        #     # print("-------------------")
        #     for swap_im in swap_stats:
        #         distance = [abs(swap_im - unswap_im) for unswap_im in unswap_stats]
        #         # print(len(distance))
        #         index = distance.index(min(distance))
        #         # print(distance[index])
        #         # print(index)
        #         # swap_law3.append(index)
        #         swap_law2.append((index-(swap_range//2))/swap_range)
        #     # print(swap_law3)
        #     # print("-------------------")
        #     img_swap = self.totensor(img_swap)
        #     label = self.labels[item] -1
        #     if self.use_cls_mul:
        #         label_swap = label + self.numcls
        #     if self.use_cls_2:
        #         label_swap = -1
        #     img_unswap = self.totensor(img_unswap)
        #     return img_unswap, img_swap, label, label_swap, swap_law1, swap_law2, swap_law3, self.paths[item]

    def jigsaw_generator_dcl(self,images, n):
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

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    def crop_image(self, image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list

    def add_mask(self,image, cropnum):
        channel, high, width = image.shape
        crop_w = [int((width / cropnum[1]) * i) for i in range(cropnum[0] + 1)]
        crop_h = [int((high / cropnum[0]) * i) for i in range(cropnum[1] + 1)]
        index = 0
        for j in range(len(crop_h) - 1):
            for i in range(len(crop_w) - 1):
                w_start = crop_w[i]
                w_end = min(crop_w[i + 1], width)
                h_start = crop_h[j]
                h_end = min(crop_h[j + 1], high)
                image[3:, h_start:h_end, w_start:w_end] = index
                index = index + 1
        return image

    def crop_np_image(self,image, cropnum):
        channel, high, width = image.shape
        crop_w = [int((width / cropnum[1]) * i) for i in range(cropnum[0] + 1)]
        crop_h = [int((high / cropnum[0]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_h) - 1):
            for i in range(len(crop_w) - 1):
                w_start = crop_w[i]
                w_end = min(crop_w[i + 1], width)
                h_start = crop_h[j]
                h_end = min(crop_h[j + 1], high)
                im_list.append(image[:, h_start:h_end, w_start:w_end])
                # image[:, crop_h[j]: min(crop_h[j + 1], high), crop_w[i] : min(crop_w[i + 1], width) ]
                # im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list

    def get_index(self,img):
        index = []
        img_np = np.asarray(img)
        img_np = np.transpose(img_np, (2, 0, 1))
        mask = np.zeros((1, 448, 448), dtype=int)

        img_np = np.concatenate((img_np, mask), axis=0)
        crop = [7, 7]
        img_np = self.add_mask(img_np, crop)
        images = self.crop_np_image(img_np, crop)

        pro = 5
        if pro >= 5:
            tmpx = []
            tmpy = []
            count_x = 0
            count_y = 0
            k = 1
            RAN = 2
            for i in range(crop[1] * crop[0]):
                tmpx.append(images[i])
                count_x += 1
                if len(tmpx) >= k:
                    tmp = tmpx[count_x - RAN:count_x]
                    random.shuffle(tmp)
                    tmpx[count_x - RAN:count_x] = tmp
                if count_x == crop[0]:
                    tmpy.append(tmpx)
                    count_x = 0
                    count_y += 1
                    tmpx = []
                if len(tmpy) >= k:
                    tmp2 = tmpy[count_y - RAN:count_y]
                    random.shuffle(tmp2)
                    tmpy[count_y - RAN:count_y] = tmp2
            random_im = []
            for line in tmpy:
                random_im.extend(line)
            for i in range(49):
                index.append(random_im[i][3][0][0])
        return index
# def collate_fn4train(batch):
#     imgs = []
#     label = []
#     img_name = []
#     for sample in batch:
#         imgs.append(sample[0])
#         label.append(sample[1])
#         img_name.append(sample[-1])
#     return torch.stack(imgs, 0), label, img_name



def collate_fn4test(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[-2])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, img_name

def collate_fn4train(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[-2])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, img_name

def collate_fn4train_withdcl(batch):
    imgs = []
    label = []
    img_name = []
    dcl_imgs = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[-2])
        img_name.append(sample[-1])
        dcl_imgs.append(sample[3])
    return torch.stack(imgs, 0), label, img_name ,torch.stack(dcl_imgs, 0)



def collate_fn4traindcl(batch):
    # img_unswap, img_swap, unswap_law, swap_law, label, self.paths[item]
    # return img_unswap, img_swap_2, img_swap_4, img_swap_8, unswap_law_2, unswap_law_8, unswap_law_8, swaplaw_2, swaplaw_4, swaplaw_8, label,  self.paths[item]
    imgs = []
    unswap_law2 = []
    unswap_law4 = []
    unswap_law8 = []
    swap_law2 = []
    swap_law4 = []
    swap_law8 = []
    img_2 = []
    img_4 = []
    img_8 = []
    label_swap = []
    label = []
    swap_label = []
    img_name = []
    unswap_label = []
    for sample in batch:
        # print("start")
        imgs.append(sample[0])
        # 2*2  first unswap ,second swap
        # img_2.append(sample[0])
        img_2.append(sample[1])
        #
        # img_4.append(sample[0])
        img_4.append(sample[2])

        # img_8.append(sample[0])
        img_8.append(sample[3])
        # first unswaplaw ,second swaplaw
        unswap_law2.append(sample[4])
        swap_law2.append(sample[7])

        unswap_law4.append(sample[5])
        swap_law4.append(sample[8])

        unswap_law8.append(sample[6])
        swap_law8.append(sample[9])

        label.append(sample[-2])
        swap_label.append(sample[-2])
        # unswap_label.append(sample[-2])
        img_name.append(sample[-1])
        # unswap 1 , swap 0
        label_swap.append(1)
        # label_swap.append(0)
        # print("over")
    img_unswap = torch.stack(imgs, 0)
    img_2  = torch.stack(img_2, 0)
    img_4  = torch.stack(img_4, 0)
    img_8  = torch.stack(img_8, 0)
    return img_unswap, img_2 ,img_4 ,img_8 , unswap_law2, unswap_law4 , unswap_law8, swap_law2, swap_law4 , swap_law8,  label, swap_label,label_swap, img_name





# def collate_fn4trainres(batch):
#     imgs = []
#     label = []
#     img_name = []
#     for sample in batch:
#         imgs.append(sample[0])
#         label.append(sample[2])
#         img_name.append(sample[-1])
#     return torch.stack(imgs, 0), label,  img_name
#
# def collate_fn4trainrcm(batch):
#     imgs = []
#     label = []
#     label_swap = []
#     law_swap = []
#     img_name = []
#     #模拟dcl的打乱方式
#     law_simu_dcl = []
#     for sample in batch:
#         imgs.append(sample[0])
#         label.append(sample[2])
#         if sample[3] == -1:
#             label_swap.append(1)
#         law_swap.append(sample[4])
#         img_name.append(sample[-1])
#         law_simu_dcl.append(sample[-2])
#     return torch.stack(imgs, 0), label, label_swap, law_swap, img_name , law_simu_dcl
#
# def collate_fn4train(batch):
#     imgs = []
#     label = []
#     label_swap = []
#     law_swap = []
#     img_name = []
#     for sample in batch:
#         imgs.append(sample[0])
#         imgs.append(sample[1])
#         label.append(sample[2])
#         label.append(sample[2])
#         if sample[3] == -1:
#             label_swap.append(1)
#             label_swap.append(0)
#         else:
#             label_swap.append(sample[2])
#             label_swap.append(sample[3])
#         law_swap.append(sample[4])
#         law_swap.append(sample[5])
#         img_name.append(sample[-1])
#     return torch.stack(imgs, 0), label, label_swap, law_swap, img_name



