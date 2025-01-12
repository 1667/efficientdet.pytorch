# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'efficientdet.pytorch'))
	print(os.getcwd())
except:
	pass
# %% [markdown]
# # setup dataset

# %%
# import stuff
import os
import numpy as np
import time
import pandas as pd

import torch
import torch.utils.data as data
from itertools import product as product

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
from utils.to_fp16 import network_to_half


# %%
# import dataset
from utils.dataset import VOCDataset, DatasetTransform, make_datapath_list, Anno_xml2list, od_collate_fn


# %%
## meta settings

# select from efficientnet backbone or resnet backbone
backbone = "efficientnet-b2"
scale = 2
# scale==1: resolution 300
# scale==2: resolution 600
useBiFPN = True
HALF = False # enable FP16
DATASET = "VOC"
retina = False # for trying retinanets

# %% [markdown]
# ## make data.Dataset for training

# %%
if not DATASET == "COCO":
    # load files
    # set your VOCdevkit path here.
    vocpath = "./VOC2007"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(vocpath)
    
    # vocpath = "../VOCdevkit/VOC2012"
    # train_img_list2, train_anno_list2, _, _ = make_datapath_list(vocpath)

    # train_img_list.extend(train_img_list2)
    # train_anno_list.extend(train_anno_list2)

    print("trainlist: ", len(train_img_list))
    print("vallist: ", len(val_img_list))

    # make Dataset
    # voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
    #                'bottle', 'bus', 'car', 'cat', 'chair',
    #                'cow', 'diningtable', 'dog', 'horse',
    #                'motorbike', 'person', 'pottedplant',
    #                'sheep', 'sofa', 'train', 'tvmonitor']

    voc_classes = ['fire']

    color_mean = (104, 117, 123)  # (BGR)の色の平均値
    if scale == 1:
        input_size = 300  # 画像のinputサイズを300×300にする
    else:
        input_size = 512

    ## DatasetTransformを適応
    transform = DatasetTransform(input_size, color_mean)
    transform_anno = Anno_xml2list(voc_classes)

    # Dataloaderに入れるデータセットファイル。
    # ゲットで叩くと画像とGTを前処理して出力してくれる。
    train_dataset = VOCDataset(train_img_list, train_anno_list, phase = "train", transform=transform, transform_anno = transform_anno)
    val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DatasetTransform(
        input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

else:
    from dataset.coco import COCODetection
    import torch.utils.data as data
    from utils.dataset import VOCDataset, COCODatasetTransform, make_datapath_list, Anno_xml2list, od_collate_fn

    color_mean = (104, 117, 123)  # (BGR)の色の平均値
    if scale == 1:
        input_size = 300  # 画像のinputサイズを300×300にする
    else:
        input_size = 512

    ## DatasetTransformを適応
    transform = COCODatasetTransform(input_size, color_mean)
    train_dataset = COCODetection("../data/coco/", image_set="train2014", phase="train", transform=transform)
    val_dataset = COCODetection("../data/coco/", image_set="val2014", phase="val", transform=transform)
    
batch_size = 32

train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn, num_workers=8)

val_dataloader = data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn, num_workers=8)

# 辞書型変数にまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}


# %%
# 動作の確認
batch_iterator = iter(dataloaders_dict["val"])  # イタレータに変換
images, targets = next(batch_iterator)  # 1番目の要素を取り出す
print(images.size())  # torch.Size([4, 3, 300, 300])
print(len(targets))
print(targets[1].shape)  # ミニバッチのサイズのリスト、各要素は[n, 5]、nは物体数

# %% [markdown]
# # define EfficientDet model

# %%
from utils.efficientdet import EfficientDet


# %%
if not DATASET == "COCO":
    num_class = 2
else:
    num_class = 81

if scale==1:
    ssd_cfg = {
        'num_classes': num_class,  # 背景クラスを含めた合計クラス数
        'input_size': 300*scale,  # 画像の入力サイズ
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
        'feature_maps': [37, 18, 9, 5, 3, 1],  # 各sourceの画像サイズ
        'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
        'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
        'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }
elif scale==2:
    ssd_cfg = {
        'num_classes': num_class,  # 背景クラスを含めた合計クラス数
        'input_size': 512,  # 画像の入力サイズ
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
        'feature_maps': [64, 32, 16, 8, 4, 2],  # 各sourceの画像サイズ
        'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
        'min_sizes': [30, 60, 111, 162, 213, 264]*scale,  # DBOXの大きさを決める
        'max_sizes': [60, 111, 162, 213, 264, 315]*scale,  # DBOXの大きさを決める
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }

# test if net works
net = EfficientDet(phase="train", cfg=ssd_cfg, verbose=True, backbone=backbone, useBiFPN=useBiFPN)
out = net(torch.rand([1,3,input_size,input_size]))
print(out[0].size())


# %%
net = EfficientDet(phase="train", cfg=ssd_cfg, verbose=False, backbone=backbone, useBiFPN=useBiFPN)

if retina:
    from utils.retinanet import RetinaFPN
    ssd_cfg = {
        'num_classes': num_class,  # 背景クラスを含めた合計クラス数
        'input_size': 300*scale,  # 画像の入力サイズ
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
        'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
        'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
        'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
        'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }
    net = RetinaFPN("train", ssd_cfg)

# GPUが使えるか確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using:", device)

print("set weights!")


# %%
# FP16..
if HALF:
    net = network_to_half(net)


# %%
print(net)


# %%
from utils.ssd_model import MultiBoxLoss

# define loss
criterion = MultiBoxLoss(jaccard_thresh=0.5,neg_pos=3, device=device, half=HALF)

# optim
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)


# %%
def get_current_lr(epoch):
    
    if DATASET == "COCO":
        reduce = [20, 40]
        # warmup
        if epoch < 1:
            lr = 1e-4
        else:
            lr = 1e-3
    else:
        reduce = [120,180]
        lr = 1e-3
        
    for i,lr_decay_epoch in enumerate(reduce):
        if epoch >= lr_decay_epoch:
            lr *= 0.1
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    print("lr is:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# %%
# モデルを学習させる関数を作成
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("used device:", device)

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # イテレーションカウンタをセット
    iteration = 1
    epoch_train_loss = 0.0  # epochの損失和
    epoch_val_loss = 0.0  # epochの損失和
    logs = []

    # epochのループ
    for epoch in range(num_epochs+1):
        
        adjust_learning_rate(optimizer, epoch)
        
        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
                print('(train)')
            else:
                if((epoch+1) % 10 == 0):
                    net.eval()   # モデルを検証モードに
                    print('-------------')
                    print('(val)')
                else:
                    # 検証は10回に1回だけ行う
                    continue

            # データローダーからminibatchずつ取り出すループ
            for images, targets in dataloaders_dict[phase]:

                # GPUが使えるならGPUにデータを送る
                images = images.to(device)
                targets = [ann.to(device)
                           for ann in targets]  # リストの各要素のテンソルをGPUへ
                if HALF:
                    images = images.half()
                    targets = [ann.half() for ann in targets]
                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    # 順伝搬（forward）計算
                    outputs = net(images)
                    #print(outputs[0].type())
                    # 損失の計算
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()  # 勾配の計算

                        # 勾配が大きくなりすぎると計算が不安定になるので、clipで最大でも勾配2.0に留める
                        nn.utils.clip_grad_value_(
                            net.parameters(), clip_value=2.0)

                        optimizer.step()  # パラメータ更新

                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('Iter {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                                iteration, loss.item(), duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1

                    # 検証時
                    else:
                        epoch_val_loss += loss.item()

        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
            epoch+1, epoch_train_loss, epoch_val_loss))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        # ログを保存
        log_epoch = {'epoch': epoch+1,
                     'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")

        epoch_train_loss = 0.0  # epochの損失和
        epoch_val_loss = 0.0  # epochの損失和

        # ネットワークを保存する
        if ((epoch+1) % 10 == 0):
            if useBiFPN:
                word="BiFPN"
            else:
                word="FPN"
            torch.save(net.state_dict(), 'weights/'+DATASET+"_"+backbone+"_" + str(300*scale) + "_" + word + "_" + 
                       str(epoch+1) + '.pth')


# %%
if DATASET == "COCO":
    num_epochs = 30
else:
    num_epochs = 200
    
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)


# %%


