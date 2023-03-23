import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import utils
from torchsummary import summary
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
import os
import copy
import numpy as np
import pandas as pd
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = [
    "milkbox"
]

class VOCDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, transform=None, trans_params=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.trans_params = trans_params

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) # /PASCAL_VOC/labels/000009.txt
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0]) # /PASCAL_VOC/images/000009.jpg
        image = np.array(Image.open(img_path).convert("RGB")) # albumentation을 적용하기 위해 np.array로 변환합니다.

        labels = None
        if os.path.exists(label_path):
            # np.roll: (class, cx, cy, w, h) -> (cx, cy, w, h, class)
            # np.loadtxt: txt 파일에서 data 불러오기
            labels = np.array(np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist())
            # labels = np.loadtxt(label_path).reshape(-1, 5)

        if self.transform:
            # apply albumentations
            augmentations = self.transform(image=image, bboxes=labels)
            image = augmentations['image']
            targets = augmentations['bboxes']
            
            # for DataLoader
            # lables: ndarray -> tensor
            # dimension: [batch, cx, cy, w, h, class]
            if targets is not None:
                target = torch.zeros((len(labels), 6))
                target[:, 1:] = torch.tensor(targets) 
        else:
            target = labels


        return image, target, label_path

train_csv_file = 'milkbox/train/train.csv'
label_dir = 'milkbox/train/labels'
img_dir = 'milkbox/train/images'

train_ds = VOCDataset(train_csv_file, img_dir, label_dir)

val_csv_file = 'milkbox/vaild/vaild.csv'
label_dir = 'milkbox/vaild/labels'
img_dir = 'milkbox/vaild/images'

val_ds = VOCDataset(val_csv_file, img_dir, label_dir)

# transforms 정의하기
IMAGE_SIZE = 416
scale = 1.0

# for train
train_transforms = A.Compose([
        # 이미지의 maxsize를 max_size로 rescale합니다. aspect ratio는 유지합니다.
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        # min_size보다 작으면 pad
        A.PadIfNeeded(min_height=int(IMAGE_SIZE * scale), min_width=int(IMAGE_SIZE * scale), border_mode=cv2.BORDER_CONSTANT),
        # random crop
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        # brightness, contrast, saturation을 무작위로 변경합니다.
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        # transforms 중 하나를 선택해 적용합니다.
        A.OneOf([
                 # shift, scale, rotate 를 무작위로 적용합니다.
                 A.ShiftScaleRotate(rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        ], p=1.0),
        # 수평 뒤집기
        A.HorizontalFlip(p=0.5),
        # blur
        A.Blur(p=0.1),
        # normalize
        A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255),
        ToTensorV2()
        ],
        # (x1, y1, x2, y2) -> (cx, cy, w, h)
        bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, label_fields=[])
        )

# for validation
val_transforms = A.Compose([
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(min_height=int(IMAGE_SIZE * scale), min_width=int(IMAGE_SIZE * scale), border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, label_fields=[])
        )

train_ds.transform = train_transforms
val_ds.transform = val_transforms

def collate_fn(batch):
    imgs, targets, paths = list(zip(*batch))
    # 빈 박스 제거하기
    targets = [boxes for boxes in targets if boxes is not None]
    # index 설정하기
    for b_i, boxes in enumerate(targets):
        boxes[:, 0] = b_i
    targets = torch.cat(targets, 0)
    imgs = torch.stack([img for img in imgs])
    return imgs, targets, paths

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)

path2config = 'yolov3-voc.cfg'

# config 파일을 분석하는 함수를 정의합니다.
def parse_model_config(path2file):
    # cfg 파일 열기
    cfg_file = open(path2file, 'r')
    # 문자열 데이터 읽어오기 
    lines = cfg_file.read().split('\n') #['[net]', '# Testing', '# batch=1', '....' ]

    # 데이터 전처리
    # startswith('#'): 문자열이 # 로 시작하는지 여부를 알려줍니다. 
    lines = [x for x in lines if x and not x.startswith('#')] # ['[net]', 'batch=64', '...']
    # 공백 제거
    lines = [x.rstrip().lstrip() for x in lines]

    blocks_list = []
    for line in lines:
        if line.startswith('['): # [net]
            blocks_list.append({}) # {}
            blocks_list[-1]['type'] = line[1:-1].rstrip() # [{'type': 'net'}]
        else:
            key, value = line.split('=') # batch=64 -> batch, 64
            value = value.strip() # 공백 제거
            blocks_list[-1][key.rstrip()] = value.strip() # 'batch':'64'

    return blocks_list

blocks_list = parse_model_config(path2config)

class EmptyLayer(nn.Module):
    def __init__(self):
        super().__init__()



class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim=416):
        super().__init__()
        self.anchors = anchors # three anchor per YOLO layer
        self.num_anchors = len(anchors) # 3
        self.num_classes = num_classes
        self.img_dim = img_dim

    def forward(self, x):
        # x: batch_size, channels, H, W
        batch_size = x.size(0)
        grid_size = x.size(2) # S = 13 or 26 or 52
        device = x.device

        prediction = x.view(batch_size, self.num_anchors, self.num_classes + 5, 
                            grid_size, grid_size) # shape = (batch, 3, 6, S, S)

        # (batch, 3, 6, S, S) -> (batch, 3, S, S, 6)
        prediction = prediction.permute(0, 1, 3, 4, 2)
        prediction = prediction.contiguous()

        obj_score = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        stride = self.img_dim / grid_size 

        # grid_x = torch.arange(grid_size, device=device).repeat(1, 1, grid_size, 1).type(torch.float32)
        # grid_y = torch.arange(grid_size, device=device).repeat(1, 1, grid_size, 1).transpose(3, 2).type(torch.float32)

        grid_x = torch.arange(grid_size, dtype=torch.float, device=device).repeat(grid_size, 1).view(
            [1, 1, grid_size, grid_size])
        grid_y = torch.arange(grid_size, dtype=torch.float, device=device).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size])

        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors] 

        scaled_anchors = torch.tensor(scaled_anchors, device=device)

        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        x = torch.sigmoid(prediction[..., 0]) # sigmoid(box x), 예측값을 sigmoid로 감싸서 [0~1] 범위
        y = torch.sigmoid(prediction[..., 1]) # sigmoid(box y), 예측값을 sigmoid로 감싸서 [0~1] 범위
        w = prediction[..., 2] # 예측한 바운딩 박스 너비
        h = prediction[..., 3]

        pred_boxes = torch.zeros_like(prediction[..., :4]).to(device)
        pred_boxes[..., 0] = x + grid_x # sigmoid(box x) + cell x 좌표
        pred_boxes[..., 1] = y + grid_y # sigmoid(box y) + cell y 좌표
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h

        # batch, num_anchor x S x S, 25
        # ex) at 13x13 -> [batch, 507, 25], at 26x26 -> [batch, 2028, 85], at 52x52 -> [batch, 10647, 85]
        # 최종적으로 YOLO는 10647개의 바운딩박스를 예측합니다.
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * stride,
                            obj_score.view(batch_size, -1, 1),
                            pred_cls.view(batch_size, -1, self.num_classes)), -1)
        #output=(배치사이즈,그리드사이즈*그리드사이즈*3,바운딩박스좌표+존재확률+어떤클래스에 존재하는 확률)
        return output
    
def create_layers(blocks_list):
    hyperparams = blocks_list[0]
    channels_list = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()

    for layer_ind, layer_dict in enumerate(blocks_list[1:]):
        modules = nn.Sequential()

        if layer_dict['type'] == 'convolutional':
            filters = int(layer_dict['filters'])
            kernel_size = int(layer_dict['size'])
            pad = (kernel_size - 1) // 2
            bn = layer_dict.get('batch_normalize', 0)

            conv2d = nn.Conv2d(in_channels=channels_list[-1], out_channels=filters, kernel_size=kernel_size,
                               stride=int(layer_dict['stride']), padding=pad, bias=not bn)
            modules.add_module('conv_{0}'.format(layer_ind), conv2d)

            if bn:
                bn_layer = nn.BatchNorm2d(filters, momentum=0.6, eps=1e-5)
                modules.add_module('batch_norm_{0}'.format(layer_ind), bn_layer)
            
            if layer_dict['activation'] == 'leaky':
                activn = nn.LeakyReLU(0.1)
                modules.add_module('leky_{0}'.format(layer_ind), activn)

        elif layer_dict["type"] == "upsample":
            stride = int(layer_dict["stride"])
            upsample = nn.Upsample(scale_factor = stride)
            modules.add_module("upsample_{}".format(layer_ind), upsample) 

        elif layer_dict["type"] == "shortcut":
            backwards=int(layer_dict["from"])
            filters = channels_list[1:][backwards]
            modules.add_module("shortcut_{}".format(layer_ind), EmptyLayer())
            
        elif layer_dict["type"] == "route":
            layers = [int(x) for x in layer_dict["layers"].split(",")]
            filters = sum([channels_list[1:][l] for l in layers])
            modules.add_module("route_{}".format(layer_ind), EmptyLayer())

        elif layer_dict["type"] == "yolo":
            anchors = [int(a) for a in layer_dict["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]

            # ex) at 13x13, 'mask': '6,7,8'
            # mask는 anchors index를 의미합니다.
            # yolo layer당 3개의 anchors를 할당 합니다.
            # mask는 yolo layer feature map size에 알맞는 anchors를 설정합니다.
            mask = [int(m) for m in layer_dict["mask"].split(",")]
            
            anchors = [anchors[i] for i in mask] # 3 anchors
            
            num_classes = int(layer_dict["classes"]) # 20
            img_size = int(hyperparams["height"]) # 416 
                        
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module("yolo_{}".format(layer_ind), yolo_layer)
            
        module_list.append(modules)       
        channels_list.append(filters)

    return hyperparams, module_list        

hy_pa, m_l = create_layers(blocks_list)

class Darknet(nn.Module):
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.blocks_list = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_layers(self.blocks_list)
        self.img_size = img_size
        
    def forward(self, x):
        img_dim = x.shape[2]
        layer_outputs, yolo_outputs = [], []
        
        # blocks_list: config 파일 분석한 결과
        # module_list: blocks_list로 생성한 module
        for block, module in zip(self.blocks_list[1:], self.module_list):
            if block["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)        
                
                
            elif block["type"] == "shortcut":
                layer_ind = int(block["from"]) # -3
                x = layer_outputs[-1] + layer_outputs[layer_ind] # shortcut connection

            #  {'type': 'yolo', 'mask': '3,4,5', 'anchors': '10,13, ...}
            elif block["type"] == "yolo":
                x= module[0](x) # get yolo layer output
                yolo_outputs.append(x)
            elif block["type"] == "route": #  {'type': 'route', 'layers': '-1, 61'}
                x = torch.cat([layer_outputs[int(l_i)] 
                               for l_i in block["layers"].split(",")], 1)
            layer_outputs.append(x)
        yolo_out_cat = torch.cat(yolo_outputs, 1) # 3개의 output을 하나로 연결
        return yolo_out_cat, yolo_outputs
    
model = Darknet(path2config).to(device)

def get_loss_batch(output,targets, params_loss, opt=None):
    ignore_thres=params_loss["ignore_thres"]
    scaled_anchors= params_loss["scaled_anchors"] # 정규화된 anchor   
    mse_loss= params_loss["mse_loss"] # nn.MSELoss
    bce_loss= params_loss["bce_loss"] # nn.BCELoss, 이진 분류에서 사용
    
    num_yolos=params_loss["num_yolos"] # 3
    num_anchors= params_loss["num_anchors"] # 3
    obj_scale= params_loss["obj_scale"] # 1
    noobj_scale= params_loss["noobj_scale"] # 100

    loss = 0.0

    for yolo_ind in range(num_yolos):
        yolo_out = output[yolo_ind] # yolo_out: batch, num_boxes, class+coordinates
        batch_size, num_bbxs, _ = yolo_out.shape

        # get grid size
        gz_2 = num_bbxs/num_anchors # ex) at 13x13, 507 / 3
        grid_size=int(np.sqrt(gz_2))

        # (batch, num_boxes, class+coordinates) -> (batch, num_anchors, S, S, class+coordinates)
        yolo_out = yolo_out.view(batch_size, num_anchors, grid_size, grid_size, -1)

        pred_boxes = yolo_out[:,:,:,:,:4] # get box coordinates
        x,y,w,h = transform_bbox(pred_boxes, scaled_anchors[yolo_ind]) # cell 내에서 x,y 좌표와  
        pred_conf = yolo_out[:,:,:,:,4] # get confidence
        pred_cls_prob = yolo_out[:,:,:,:,5:]

        yolo_targets = get_yolo_targets({
            'pred_cls_prob':pred_cls_prob,
            'pred_boxes':pred_boxes,
            'targets':targets,
            'anchors':scaled_anchors[yolo_ind],
            'ignore_thres':ignore_thres,
        })

        obj_mask=yolo_targets["obj_mask"]        
        noobj_mask=yolo_targets["noobj_mask"]            
        tx=yolo_targets["tx"]                
        ty=yolo_targets["ty"]                    
        tw=yolo_targets["tw"]                        
        th=yolo_targets["th"]                            
        tcls=yolo_targets["tcls"]                                
        t_conf=yolo_targets["t_conf"]

        #x[obj_mask] obj_mask이 ture인 곳에서만 loss를 계산하기 위해서
        loss_x = mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = mse_loss(h[obj_mask], th[obj_mask])
        
        loss_conf_obj = bce_loss(pred_conf[obj_mask], t_conf[obj_mask])
        loss_conf_noobj = bce_loss(pred_conf[noobj_mask], t_conf[noobj_mask])
        loss_conf = obj_scale * loss_conf_obj + noobj_scale * loss_conf_noobj
        loss_cls = bce_loss(pred_cls_prob[obj_mask], tcls[obj_mask])
        loss += loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    return loss.item()

def transform_bbox(bbox, anchors):
    # bbox: predicted bbox coordinates
    # anchors: scaled anchors

    x = bbox[:,:,:,:,0]
    y = bbox[:,:,:,:,1]
    w = bbox[:,:,:,:,2]
    h = bbox[:,:,:,:,3]
    anchor_w = anchors[:,0].view((1,3,1,1))
    anchor_h = anchors[:,1].view((1,3,1,1))

    x=x-x.floor() # 전체 이미지의 x 좌표에서 셀 내의 x좌표로 변경
    y=y-y.floor() # 전체 이미지의 y 좌표에서 셀 내의 y좌표로 변경
    w=torch.log(w / anchor_w + 1e-16)
    h=torch.log(h / anchor_h + 1e-16)
    return x, y, w, h

def get_yolo_targets(params):
    pred_boxes = params['pred_boxes']
    pred_cls_prob = params['pred_cls_prob']
    target = params['targets'] # batchsize, cls, cx, cy, w, h
    anchors = params['anchors']
    ignore_thres = params['ignore_thres']
    #배치사이즈
    batch_size = pred_boxes.size(0)
    #앵커갯수
    num_anchors = pred_boxes.size(1)
    #13 or 26 or 52
    grid_size = pred_boxes.size(2)
    #1개
    num_cls = pred_cls_prob.size(-1)


    sizeT = batch_size, num_anchors, grid_size, grid_size
    obj_mask = torch.zeros(sizeT, device=device, dtype=torch.bool)
    noobj_mask = torch.ones(sizeT, device=device, dtype=torch.bool)
    tx = torch.zeros(sizeT, device=device, dtype=torch.float32)
    ty = torch.zeros(sizeT, device=device, dtype=torch.float32)
    tw = torch.zeros(sizeT, device=device, dtype=torch.float32)
    th = torch.zeros(sizeT, device=device, dtype=torch.float32)

    sizeT = batch_size, num_anchors, grid_size, grid_size, num_cls
    tcls = torch.zeros(sizeT, device=device, dtype=torch.float32)

    # 데이터셋에 박스를 쳐놓은 실제 바운딩박스 좌표를 의미
    # target = batch, cx, cy, w, h, class
    target_bboxes = target[:, 1:5] * grid_size
    t_xy = target_bboxes[:, :2]
    t_wh = target_bboxes[:, 2:]
    t_x, t_y = t_xy.t() # .t(): 전치
    t_w, t_h = t_wh.t() # .t(): 전치
    #중심 좌표가 위치한 셀의 행과 열 인덱스를 나타냄
    grid_i, grid_j = t_xy.long().t() # .long(): int로 변환

    # anchor와 target의 iou 계산
    iou_with_anchors = [get_iou_WH(anchor, t_wh) for anchor in anchors]
    iou_with_anchors = torch.stack(iou_with_anchors)
    best_iou_wa, best_anchor_ind = iou_with_anchors.max(0) # iou가 가장 높은 anchor 추출

    batch_inds, target_labels = target[:, 0].long(), target[:, 5].long()
    obj_mask[batch_inds, best_anchor_ind, grid_j, grid_i] = 1 # iou가 가장 높은 anchor 할당
    noobj_mask[batch_inds, best_anchor_ind, grid_j, grid_i] = 0

    # threshold 보다 높은 iou를 지닌 anchor
    # iou가 가장 높은 anchor만 할당하면 되기 때문입니다.
    #해당 앵커의 iou가 ignore_thres보다 크다면 객체가 존재한다고 생각하고 0을 작다면 없다고 생각하고 1을 할당한다.
    for ind, iou_wa in enumerate(iou_with_anchors.t()):
        noobj_mask[batch_inds[ind], iou_wa > ignore_thres, grid_j[ind], grid_i[ind]] = 0

        # cell 내에서 x,y로 변환
    tx[batch_inds, best_anchor_ind, grid_j, grid_i] = t_x - t_x.floor()
    ty[batch_inds, best_anchor_ind, grid_j, grid_i] = t_y - t_y.floor()

    anchor_w = anchors[best_anchor_ind][:, 0]
    tw[batch_inds, best_anchor_ind, grid_j, grid_i] = torch.log(t_w / anchor_w + 1e-16)

    anchor_h = anchors[best_anchor_ind][:, 1]
    th[batch_inds, best_anchor_ind, grid_j, grid_i] = torch.log(t_h / anchor_h + 1e-16)

    tcls[batch_inds, best_anchor_ind, grid_j, grid_i, target_labels] = 1

    output = {
        'obj_mask': obj_mask,
        'noobj_mask': noobj_mask,
        'tx': tx,
        'ty': ty,
        'tw': tw,
        'th': th,
        'tcls': tcls,
        't_conf': obj_mask.float(),
    }
    return output

#wh1 앵커 wh2 데이터셋 바운딩박스 wh
def get_iou_WH(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    #iou=교차영역(inter_area)/합집합영역(union_area)
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

opt = optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=60,verbose=1)

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
    
def loss_epoch(model,params_loss,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    len_data=len(dataset_dl.dataset)
    running_metrics= {}
    
    for xb, yb,_ in dataset_dl:
        yb=yb.to(device)
        _,output=model(xb.to(device))
        loss_b=get_loss_batch(output,yb, params_loss,opt)
        running_loss+=loss_b
        if sanity_check is True:
            break 
    loss=running_loss/float(len_data)
    return loss
     
import time
def train_val(model, params):
    num_epochs=params["num_epochs"]
    params_loss=params["params_loss"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]
    
    
    loss_history={
        "train": [],
        "val": [],
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf') 
    
    start_time = time.time()
    for epoch in range(num_epochs):
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr)) 
        model.train()
        train_loss=loss_epoch(model,params_loss,train_dl,sanity_check,opt)
        loss_history["train"].append(train_loss)  
        
        model.eval()
        with torch.no_grad():
            val_loss=loss_epoch(model,params_loss,val_dl,sanity_check)
        loss_history["val"].append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
            print('Get best val loss')
            
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts) 
        print("train loss: %.6f, val loss: %.6f, time: %.4f min" %(train_loss, val_loss, (time.time()-start_time)/60))
        print("-"*10) 
    model.load_state_dict(best_model_wts)
    return model, loss_history

path2models= "./models/"
if not os.path.exists(path2models):
        os.mkdir(path2models)
        
scaled_anchors=[torch.tensor([[3.6250,  2.8125],[4.8750,  6.1875],[11.6562, 10.1875]], device=device),
               torch.tensor([[1.8750, 3.8125],[3.8750, 2.8125],[3.6875, 7.4375]], device=device),
               torch.tensor([[1.2500, 1.6250],[2.0000, 3.7500],[4.1250, 2.8750]], device=device)]


mse_loss = nn.MSELoss(reduction="mean")
bce_loss = nn.BCELoss(reduction="mean")
params_loss={
    "scaled_anchors" : scaled_anchors,
    "ignore_thres": 0.5,
    "mse_loss": mse_loss,
    "bce_loss": bce_loss,
    "num_yolos": 3,
    "num_anchors": 3,
    "obj_scale": 1,
    "noobj_scale": 100,
}

params_train={
    "num_epochs": 1000,
    "optimizer": opt,
    "params_loss": params_loss,
    "train_dl": train_dl,
    "val_dl": val_dl,
    "sanity_check": False,
    "lr_scheduler": lr_scheduler,
    "path2weights": path2models+"weights.pt",
}

 
model,loss_hist=train_val(model,params_train)

# from utils import non_max_suppression, rescale_boxes

# model.load_state_dict(torch.load("models\weights.pt"))
# model.eval()

# conf_threshold = 0.7

# # set non-maximum suppression threshold
# nms_threshold = 0.7

# cap = cv2.VideoCapture(0)

# while True:
#     # read a frame from the video stream
#     ret, frame = cap.read()

#     # resize and normalize image
#     img = cv2.resize(frame, (416, 416))
#     img = img.astype(np.float32) / 255.0
#     img = np.transpose(img, (2, 0, 1))
#     img = np.expand_dims(img, axis=0)
#     img = torch.from_numpy(img).to(device)

#     # run inference
#     with torch.no_grad():
#         outputs, _ = model(img)
#         outputs = non_max_suppression(outputs, conf_threshold, nms_threshold)
#     # process outputs
#     if outputs[0] is not None:
#         outputs = outputs[0].cpu().numpy()
#         outputs = rescale_boxes(outputs, 416, frame.shape[:2][::-1])
#         for output in outputs:
#             label = classes[int(output[-1])]
#             confidence = output[-2]
#             x1, y1, x2, y2 = output[:4]
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(frame, "{} {:.2f}".format(label, confidence), (int(x1), int(y1) - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # show the output
#     cv2.imshow("Object Detection", frame)

#     # break the loop if 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # release the video capture
# cap.release()

# # close all windows
# cv2.destroyAllWindows()





# model.load_state_dict(torch.load("models\weights.pt"))
# model.eval()

# conf_threshold = 0.5

# # set non-maximum suppression threshold
# nms_threshold = 0.4

# cap = cv2.VideoCapture(0)

# while True:
#     # read a frame from the video stream
#     ret, frame = cap.read()

#     # resize and normalize image
#     input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     input_img = cv2.resize(input_img, (416, 416))
#     input_img = input_img.astype(np.float32) / 255.
#     input_img = np.transpose(input_img, (2, 0, 1))
#     input_tensor = torch.from_numpy(input_img).unsqueeze(0).to(device)

#     frame_width, frame_height = frame.shape[1], frame.shape[0]
#     x_scale = frame_width/416
#     y_scale = frame_height/416

#     # run inference
#     with torch.no_grad():
#         outputs, _ = model(input_tensor)

#     classes_ids =[]
#     confidences=[]
#     boxes = []
#     for output in outputs:
#         for detection in output:
#             scores = detection[5:]
#             scores_cpu=scores.cpu()
#             class_id = np.argmax(scores_cpu)
#             confidence = scores_cpu[class_id]
#             if confidence > 0.7:
#                 center_x = int(detection[0] * frame.shape[1])
#                 center_y = int(detection[1] * frame.shape[0])
#                 width = int(detection[2] * frame.shape[1])
#                 height = int(detection[3] * frame.shape[0])
#                 left = int(center_x - width / 2)
#                 top = int(center_y - height / 2)
#                 boxes.append([left, top, width, height])
#                 confidences.append(float(confidence))
#                 classes_ids.append(class_id)

#     # 비최대 억제(NMS) 수행하여 최종 검출된 객체 정보 추출
#     indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.7)
#     print(indices.shape)
#     for i in indices:
#         box = boxes[i]
#         left, top, width, height = box
#         label = f'{classes[classes_ids[i]]} {confidences[i]:.2f}'
#         cv2.rectangle(frame, (left, top), (left+width, top+height), (0,255,0), 2)
#         cv2.putText(frame, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

#     # show the output
#     cv2.imshow("Object Detection", frame)

#     # break the loop if 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # release the video capture
# cap.release()

# # close all windows
# cv2.destroyAllWindows()