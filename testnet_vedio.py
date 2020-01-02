# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
#   
# from IPython import get_ipython

#   
import argparse
import numpy as np
import cv2  # OpenCVライブラリ

import matplotlib.pyplot as plt 
# get_ipython().run_line_magic('matplotlib', 'inline')
import threading
import torch

import pickle

import torch.utils.data as data
from itertools import product as product

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import time,os
from utils.to_fp16 import network_to_half

#   
# import dataset
from utils.dataset import VOCDataset, COCODatasetTransform, make_datapath_list, Anno_xml2list, od_collate_fn


#   
# set your VOCdevkit path!
vocpath = "./VOC2007"
DEVKIT_PATH = "../VOCdevkit/"
SET = "test"
# train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(vocpath)

parser = argparse.ArgumentParser(
    description='Efficient Detector Training With Pytorch')

parser.add_argument('--model_file', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--backbone', default='efficientnet-b0', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--half', default=False, type=bool,
                    help='Checkpoint state_dict file to resume training from')

args = parser.parse_args()

outputdir = "./outputtest/"
os.system("rm -rf "+outputdir+"*")
picdir = "/home/grobot/mywork/firedetection/MobileNetV3-SSD-master/testpic/"
if args.model_file == None:
    print("Please input model file")
    sys.exit(0)

print("load model "+args.model_file)
model_file = args.model_file

HALF = args.half # enable FP16
pics = os.listdir(picdir)

val_img_list = pics

model="efficientdet"
backbone = args.backbone
scale = 1


#   
print(val_img_list[0])


#   
# image_index = []
# for l in val_img_list:
#     image_index.append(l[-10:-4])

# image_index[0]


# #   
# class_names = ['aeroplane', 'bicycle', 'bird', 'boat',
#                'bottle', 'bus', 'car', 'cat', 'chair',
#                'cow', 'diningtable', 'dog', 'horse',
#                'motorbike', 'person', 'pottedplant',
#                'sheep', 'sofa', 'train', 'tvmonitor']
class_names = ['fire']
color_mean = (104, 117, 123)  # (BGR)の色の平均値
input_size = 300  # 画像のinputサイズを300×300にする

## DatasetTransformを適応
#transform = DatasetTransform(input_size, color_mean)
#transform_anno = Anno_xml2list(class_names)


#   
#val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=COCODatasetTransform(
#    input_size, color_mean), transform_anno=Anno_xml2list(class_names))


#   
#val_dataloader = data.DataLoader(
#    val_dataset, batch_size=1, shuffle=False, collate_fn=od_collate_fn, num_workers=1)

#    [markdown]
# # set up model

#   
if model=="SSD":
    from utils.ssd_model import SSD
elif model=="retina":
    from utils.retinanet import RetinaFPN as SSD
    from utils.retinanet import Bottleneck
elif model=="efficientdet":
    from utils.efficientdet import EfficientDet
    
voc_classes = ['fire']
num_classes = len(voc_classes)+1
if scale==1:
    ssd_cfg = {
        'num_classes': len(voc_classes)+1,  # 背景クラスを含めた合計クラス数
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
        'num_classes': len(voc_classes)+1,  # 背景クラスを含めた合計クラス数
        'input_size': 300*scale,  # 画像の入力サイズ
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
        'feature_maps': [75, 38, 19, 10, 5, 3],  # 各sourceの画像サイズ
        'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
        'min_sizes': [30, 60, 111, 162, 213, 264]*scale,  # DBOXの大きさを決める
        'max_sizes': [60, 111, 162, 213, 264, 315]*scale,  # DBOXの大きさを決める
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }

# SSDネットワークモデル
if model=="SSD":
    net = SSD(phase="inference", cfg=ssd_cfg).eval()
    net_weights = torch.load('./weights/ssd300_200.pth',
                         map_location={'cuda:0': 'cpu'})
elif model=="retina":
    net = SSD(Bottleneck, [2,2,2,2], phase="inference", cfg=ssd_cfg, model=backbone).to("cuda")
    net_weights = torch.load('./weights/retinanet300_200.pth',
                         map_location={'cuda:0': 'cpu'})
else:
    net = EfficientDet(phase="inference", cfg=ssd_cfg, verbose=False, backbone=backbone, useBiFPN=True,half=HALF)

    if HALF:
	    net = network_to_half(net)
    net_weights = torch.load(model_file,
                         map_location={'cuda:0': 'cpu'})

net.load_state_dict(net_weights)

print('loaded the trained weights')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using:", device)

net = net.to(device)


#   
all_imgs = []
classes = {}
bbox_threshold = 0.05

# define detections
all_boxes = [[[] for _ in range(len(val_img_list))]
               for _ in range(num_classes)]
empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))


#   
from utils.ssd_predict_show import SSDPredictShow
ssd = SSDPredictShow(eval_categories=voc_classes, net=net, device=device)

#    [markdown]
# # infer images

#   
#all_boxes = ssd.ssd_inference(val_dataloader, all_boxes, data_confidence_level=bbox_threshold)


#   
val_img_list[0:10]


def save_result_image(image_path,boxes,class_name):

    orig_image = cv2.imread(image_path)

    for i in range(len(boxes)):
        box = boxes[i]
        cv2.rectangle(orig_image, (box[1], box[2]), (box[3], box[4]), (255, 255, 0), 4)
        #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        label = f"{class_name}: {box[0]:.2f}"
        cv2.putText(orig_image, label,
                    (int(box[1]) + 20, int(box[2]) + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type

    outputdir = "./outputtest/"

    path = outputdir+"output_"+str(time.time())+".jpg"
    cv2.imwrite(path, orig_image)
    print(f"Found {len(boxes)} objects. The output image is {path}")

#   

print(cv2.__version__)
videoCapture = cv2.VideoCapture('fire-video.mp4')
  
#获得码率及尺寸
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
 
 
#读帧
success, frame = videoCapture.read()
wait_time = int(1000/int(fps))
while success :

    time_start=time.time()
    detections, pre_dict_label_index = ssd.ssd_predict2(None, data_confidence_level=0.05,half=True,image_data=frame)
    for cls in range(len(class_names)):
        box = []
        for j,pred in enumerate(detections):
            if cls == pre_dict_label_index[j]:
                if pred[0] >= 0.5:
                    box.append(pred)
                    cv2.rectangle(frame, (pred[1], pred[2]), (pred[3], pred[4]), (255, 255, 0), 4)
                    #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
                    label = f"{class_names[cls]}: {pred[0]:.2f}"
                    cv2.putText(frame, label,
                                (int(pred[1]) + 20, int(pred[2]) + 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,  # font scale
                                (255, 0, 255),
                                2)  # line type
                # print(pred,j)
        # if not box == []:
        #     all_boxes[cls][i] = box
        # else:
        #     all_boxes[cls][i] = empty_array

    # time.sleep(1)
    time_end=time.time()
    spent_time = (time_end-time_start)*1000
    # print("time",spent_time)
    cv2.imshow('windows', frame) #显示
    if wait_time-spent_time > 0:
        cv2.waitKey(int(wait_time-spent_time)) #延迟
    
    success, frame = videoCapture.read() #获取下一帧

videoCapture.release()


# for i, imp in enumerate(val_img_list):
#     detections, pre_dict_label_index = ssd.ssd_predict2(picdir+imp, data_confidence_level=0.05,half=True)
    
#     for cls in range(len(class_names)):
#         box = []
#         for j,pred in enumerate(detections):
#             if cls == pre_dict_label_index[j]:
#                 if pred[0] >= 0.5:
#                     box.append(pred)
#                 # print(pred,j)
#         if not box == []:
#             all_boxes[cls][i] = box
#         else:
#             all_boxes[cls][i] = empty_array
        
#         save_result_image(picdir+imp,box,class_names[cls])

#     if i%1000==0:
#         print("iter:", i)



# print(all_boxes)
#   
# all_boxes[7][0:10]

#    [markdown]
# # eval accuracy

#   
# eval function
# def voc_eval(detpath,
#              annopath,
#              imagesetfile,
#              classname,
#              cachedir,
#              ovthresh=0.5,
#              use_07_metric=False):
#   """
#   rec, prec, ap = voc_eval(detpath,
#                               annopath,
#                               imagesetfile,
#                               classname,
#                               [ovthresh],
#                               [use_07_metric])
#   Top level function that does the PASCAL VOC evaluation.
#   detpath: Path to detections
#       detpath.format(classname) should produce the detection results file.
#   annopath: Path to annotations
#       annopath.format(imagename) should be the xml annotations file.
#   imagesetfile: Text file containing the list of images, one image per line.
#   classname: Category name (duh)
#   cachedir: Directory for caching the annotations
#   [ovthresh]: Overlap threshold (default = 0.5)
#   [use_07_metric]: Whether to use VOC07's 11 point AP computation
#       (default False)
#   """
#   # assumes detections are in detpath.format(classname)
#   # assumes annotations are in annopath.format(imagename)
#   # assumes imagesetfile is a text file with each line an image name
#   # cachedir caches the annotations in a pickle file

#   # first load gt
#   if not os.path.isdir(cachedir):
#     os.mkdir(cachedir)
#   cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)
#   # read list of images
#   with open(imagesetfile, 'r') as f:
#     lines = f.readlines()
#   imagenames = [x.strip() for x in lines]

#   if not os.path.isfile(cachefile):
#     # load annotations
#     recs = {}
#     for i, imagename in enumerate(imagenames):
#       recs[imagename] = parse_rec(annopath.format(imagename))
#       if i % 100 == 0:
#         print('Reading annotation for {:d}/{:d}'.format(
#           i + 1, len(imagenames)))
#     # save
#     #print('Saving cached annotations to {:s}'.format(cachefile))
#     #with open(cachefile, 'wb') as f:
#     #  pickle.dump(recs, f)
#   else:
#     # load
#     with open(cachefile, 'rb') as f:
#       try:
#         recs = pickle.load(f)
#       except:
#         recs = pickle.load(f, encoding='bytes')

#   # extract gt objects for this class
#   class_recs = {}
#   npos = 0
#   for imagename in imagenames:
#     R = [obj for obj in recs[imagename] if obj['name'] == classname]
#     bbox = np.array([x['bbox'] for x in R])
#     difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
#     det = [False] * len(R)
#     npos = npos + sum(~difficult)
#     class_recs[imagename] = {'bbox': bbox,
#                              'difficult': difficult,
#                              'det': det}

#   # read dets
#   detfile = detpath.format(classname)
#   with open(detfile, 'r') as f:
#     lines = f.readlines()

#   splitlines = [x.strip().split(' ') for x in lines]
#   image_ids = [x[0] for x in splitlines]
#   confidence = np.array([float(x[1]) for x in splitlines])
#   BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

#   nd = len(image_ids)
#   tp = np.zeros(nd)
#   fp = np.zeros(nd)

#   if BB.shape[0] > 0:
#     # sort by confidence
#     sorted_ind = np.argsort(-confidence)
# #    sorted_scores = np.sort(-confidence)
#     BB = BB[sorted_ind, :]
#     image_ids = [image_ids[x] for x in sorted_ind]

#     # go down dets and mark TPs and FPs
#     for d in range(nd):
#       id = image_ids[d][-10:-4]
#       #print(id)
#       # catch bad detections
#       try:
#           R = class_recs[id]
#       except:
#         #print("det not found")
#         continue
        
#       bb = BB[d, :].astype(float)
#       ovmax = -np.inf
#       BBGT = R['bbox'].astype(float)

#       if BBGT.size > 0:
#         # compute overlaps
#         # intersection
#         ixmin = np.maximum(BBGT[:, 0], bb[0])
#         iymin = np.maximum(BBGT[:, 1], bb[1])
#         ixmax = np.minimum(BBGT[:, 2], bb[2])
#         iymax = np.minimum(BBGT[:, 3], bb[3])
#         iw = np.maximum(ixmax - ixmin + 1., 0.)
#         ih = np.maximum(iymax - iymin + 1., 0.)
#         inters = iw * ih

#         # union
#         uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
#                (BBGT[:, 2] - BBGT[:, 0] + 1.) *
#                (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

#         overlaps = inters / uni
#         ovmax = np.max(overlaps)
#         jmax = np.argmax(overlaps)

#       if ovmax > ovthresh:
#         if not R['difficult'][jmax]:
#           if not R['det'][jmax]:
#             tp[d] = 1.
#             R['det'][jmax] = 1
#           else:
#             fp[d] = 1.
#       else:
#         fp[d] = 1.

#   # compute precision recall
#   fp = np.cumsum(fp)
#   tp = np.cumsum(tp)
#   rec = tp / float(npos)
#   # avoid divide by zero in case the first detection matches a difficult
#   # ground truth
#   prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
#   ap = voc_ap(rec, prec, use_07_metric)

#   return rec, prec, ap


# #   
# pascal_classes = np.asarray(['aeroplane', 'bicycle', 'bird', 'boat',
#                'bottle', 'bus', 'car', 'cat', 'chair',
#                'cow', 'diningtable', 'dog', 'horse',
#                'motorbike', 'person', 'pottedplant',
#                'sheep', 'sofa', 'train', 'tvmonitor'])
# PASCAL_CLASSES = pascal_classes

# #    [markdown]
# # # write out detections for evaluation

# #   
# import os 
# def get_voc_results_file_template(cls):
#         # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
#         filename = 'det_' + "val" + '_'+cls+'.txt'
#         filedir = os.path.join(DEVKIT_PATH, 'results', 'VOC2007', 'Main')
#         if not os.path.exists(filedir):
#             os.makedirs(filedir)
#         path = os.path.join(filedir, filename)
#         return path


# def write_voc_results_file(pascal_classes, all_boxes, image_index):
#         for cls_ind, cls in enumerate(pascal_classes):
#             if cls == '__background__':
#                 continue
#             print('Writing {} VOC results file'.format(cls))
#             filename = get_voc_results_file_template(cls)
#             with open(filename, 'wt') as f:
#                 for im_ind, index in enumerate(image_index):
#                     dets = np.asarray(all_boxes[cls_ind][im_ind])
#                     if dets == []:
#                         continue
#                     # the VOCdevkit expects 1-based indices
#                     for k in range(dets.shape[0]):
#                         #print(dets[k, 0])
#                         f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
#                                 format(index, dets[k, 0],
#                                        dets[k, 1] + 1, dets[k, 2] + 1,
#                                        dets[k, 3] + 1, dets[k, 4] + 1))
# import xml.etree.ElementTree as ET
# def parse_rec(filename):
#   """ Parse a PASCAL VOC xml file """
#   tree = ET.parse(filename)
#   objects = []
#   for obj in tree.findall('object'):
#     obj_struct = {}
#     obj_struct['name'] = obj.find('name').text
#     obj_struct['pose'] = obj.find('pose').text
#     obj_struct['truncated'] = int(obj.find('truncated').text)
#     obj_struct['difficult'] = int(obj.find('difficult').text)
#     bbox = obj.find('bndbox')
#     obj_struct['bbox'] = [int(bbox.find('xmin').text),
#                           int(bbox.find('ymin').text),
#                           int(bbox.find('xmax').text),
#                           int(bbox.find('ymax').text)]
#     objects.append(obj_struct)

#   return objects
# def voc_ap(rec, prec, use_07_metric=False):
#   """ ap = voc_ap(rec, prec, [use_07_metric])
#   Compute VOC AP given precision and recall.
#   If use_07_metric is true, uses the
#   VOC 07 11 point method (default:False).
#   """
#   if use_07_metric:
#     # 11 point metric
#     ap = 0.
#     for t in np.arange(0., 1.1, 0.1):
#       if np.sum(rec >= t) == 0:
#         p = 0
#       else:
#         p = np.max(prec[rec >= t])
#       ap = ap + p / 11.
#   else:
#     # correct AP calculation
#     # first append sentinel values at the end
#     mrec = np.concatenate(([0.], rec, [1.]))
#     mpre = np.concatenate(([0.], prec, [0.]))

#     # compute the precision envelope
#     for i in range(mpre.size - 1, 0, -1):
#       mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

#     # to calculate area under PR curve, look for points
#     # where X axis (recall) changes value
#     i = np.where(mrec[1:] != mrec[:-1])[0]

#     # and sum (\Delta recall) * prec
#     ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
#   return ap


# #   
# get_ipython().system('rm ../VOCdevkit/results/VOC2007/Main/*')


# #   
# write_voc_results_file(pascal_classes, all_boxes, val_img_list)

# #    [markdown]
# # # evaluation

# #   
# def python_eval(output_dir='output'):
#         annopath = os.path.join(
#             DEVKIT_PATH,
#             'VOC2007',
#             'Annotations',
#             '{:s}.xml')
#         imagesetfile = os.path.join(
#             DEVKIT_PATH,
#             'VOC2007',
#             'ImageSets',
#             'Main',
#             SET + '.txt')
#         cachedir = os.path.join(DEVKIT_PATH, 'annotations_cache')
#         aps = []
#         # The PASCAL VOC metric changed in 2010.
#         # VOC07 metric is quite old so don't use.
#         use_07_metric = False
#         print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
#         if not os.path.isdir(output_dir):
#             os.mkdir(output_dir)
#         for i, cls in enumerate(PASCAL_CLASSES):
#             if cls == '__background__':
#                 continue
#             filename = get_voc_results_file_template(cls)
#             rec, prec, ap = voc_eval(
#                 filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
#                 use_07_metric=use_07_metric)
#             aps += [ap]
#             print('AP for {} = {:.4f}'.format(cls, ap))
#             with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
#                 pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
#         print('Mean AP = {:.4f}'.format(np.mean(aps)))
#         print('~~~~~~~~')
#         print('Results:')
#         for ap in aps:
#             print('{:.3f}'.format(ap))
#         print('{:.3f}'.format(np.mean(aps)))
#         print('~~~~~~~~')
#         print('')
#         print('--------------------------------------------------------------')
#         print('Results computed with the **unofficial** Python eval code.')
#         print('Results should be very close to the official MATLAB eval code.')
#         print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
#         print('-- Thanks, The Management')
#         print('--------------------------------------------------------------')


# #   
# # evaluate detections
# python_eval()


# #   


