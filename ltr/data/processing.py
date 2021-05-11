import os
import random
import time
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
import pydicom
import torch
import torchvision.transforms as transforms
from scipy import signal
from visdom import Visdom

from ltr.data import augmentation
from ltr.data.augmentation import Rotate
from pydoctor import TensorDict
import ltr.data.processing_utils as prutils
import numpy as np

def stack_tensors(x):
    if isinstance(x,(list,tuple)) and isinstance(x[0],torch.Tensor):
        return torch.stack(x)
    return x


def rotate_bound(image, angle):
    # 获取图像的尺寸
    # 旋转中心
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    # 设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像旋转后的新边界
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    return cv2.warpAffine(image, M, (nW, nH))

class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
         through the network. For example, it can be used to crop a search region around the object, apply various data
         augmentations, etc."""
    def __init__(self,transform=transforms.ToTensor(),train_transform=None,test_transform=None,joint_transform=None):
        """
               args:
                   transform       - The set of transformations to be applied on the images. Used only if train_transform or
                                       test_transform is None.
                   train_transform - The set of transformations to be applied on the train images. If None, the 'transform'
                                       argument is used instead.
                   test_transform  - The set of transformations to be ap plied on the test images. If None, the 'transform'
                                       argument is used instead.
                   joint_transform - The set of transformations to be applied 'jointly' on the train and test images.  For
                                       example, it can be used to convert both test and train images to grayscale.
               """
        self.transform = {'train': transform if train_transform is None else train_transform,
                          'test': transform if test_transform is None else test_transform,
                          'joint': joint_transform}

    def __call__(self, data:TensorDict):
        raise NotImplementedError


class IOUProcessing():
    pass
class CropProcessing():
    def __init__(self,resize_sz,block_mode,scale_patch,train_mode=True,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.resize_sz = resize_sz
        self.mode = block_mode
        self.scale_patch = scale_patch
        self.train_mode = train_mode
        self.shift = True


    def _generate_bbox_label(self, label, bbox_size):
        bbox_list = []
        for key, value in label.items():
            x_coord,y_coord = value['coord'][0],value['coord'][1]
            bbox = [y_coord - bbox_size[0]//2,x_coord - bbox_size[1]//2,y_coord + bbox_size[0]//2,x_coord + bbox_size[1]//2]
            # if self.shift:
            #     bbox = [float(x) for x in bbox]
            #     bbox = prutils.perturb_box(torch.tensor(bbox),min_iou=0.9,sigma_factor=0.1)[0].tolist()
            #     bbox = [int(x) for x in bbox]
            bbox_list.append({'bbox':bbox,'cls':value['class_num']})
        return bbox_list


    def __call__(self,data:TensorDict):
        if self.mode == 'body':
            bbox_size = prutils.generate_bbox_size(data=data,scale=self.scale_patch)
            bbox_list = self._generate_bbox_label(data['body_anno'],bbox_size)
        elif self.mode == 'disc':
            bbox_size = prutils.generate_bbox_size(data=data, scale=self.scale_patch)
            bbox_list = self._generate_bbox_label(data['disc_anno'], bbox_size)
        index = random.sample(range(0,len(bbox_list)),k=3)
        bbox_dict = [bbox_list[idx] for idx in index]
        patch_list,label_list = [],[]
        for bbox_index in bbox_dict:
            bbox = bbox_index['bbox']
            class_name = torch.tensor(bbox_index['cls'])
            patch = ((data['train_image'][:,bbox[0]:bbox[2],bbox[1]:bbox[3]])/255.0)

            if self.train_mode:
                # random rotate
                angle_list = range(-30,30,1)
                angle = random.sample(angle_list,k=1)[0]
                patch = (rotate_bound(patch.transpose(1, 2, 0),angle)).transpose(2,0,1)
            # # save the image for vis
            # plt.figure(1)
            # plt.subplot(1,2,1)
            # plt.imshow(patch[0])
            # plt.subplot(1, 2, 2)
            # plt.imshow(new_frame[0])
            # plt.savefig('/home/adminer/SPARK/PyDoctor/ltr/workspace/plt/'+str(time.time())+'.png')
            # plt.show()
            train_patch = prutils.resize_patch(patch,self.resize_sz)
            patch_list.append(train_patch)
            label_list.append(class_name)

        train_info_dict = TensorDict({'train_image':patch_list,
                                      'train_anno':label_list})

        train_info_dict = train_info_dict.apply(stack_tensors)
        return train_info_dict



class LocProcessing():
    def __init__(self,resize_sz,block_mode,train_mode,guass_params,arguement_params,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.resize_sz = resize_sz
        self.mode = block_mode
        self.train_mode = train_mode
        self.guass_params = guass_params
        self.argumentation = arguement_params

    def _generate_guass_label(self, image, label, kernel):
        shape = image.shape
        loc_mask = np.zeros((shape[1], shape[2]), dtype=np.float32)
        for key, value in label.items():
            loc_mask[value['coord'][1]][value['coord'][0]] = 1
        loc_mask = signal.convolve2d(loc_mask, kernel, boundary='symm', mode='same')
        return loc_mask


    def __call__(self,data:TensorDict):
        if self.mode == 'body':
            kernel = prutils.generate_guass_kernel(data=data,scale_factor=self.guass_params['scale'],sigma_factor=self.guass_params['sigma'])
            guass_mask_label = self._generate_guass_label(data['train_image'],data['body_anno'],kernel)
        elif self.mode == 'disc':
            kernel = prutils.generate_guass_kernel(data=data,scale_factor=self.guass_params['scale'],sigma_factor=self.guass_params['sigma'])
            guass_mask_label = self._generate_guass_label(data['train_image'],data['disc_anno'],kernel)

        frame = data['train_image']/255.0
        label = guass_mask_label

        # train_frame,label = prutils.resize_data(frame,label,self.resize_sz)
        if self.train_mode:
            angle_list = range(self.argumentation['rotate_angle'][0],self.argumentation['rotate_angle'][1],1)
            angle = random.sample(angle_list,k=1)[0]
            frame = rotate_bound(frame.transpose(1, 2, 0),angle).transpose(2,0,1)
            label = (rotate_bound(np.expand_dims(label,0).repeat(3,axis=0).transpose(1, 2, 0),angle).transpose(2,0,1))[0]

        train_frame, label = prutils.resize_data(frame, label, self.resize_sz)
        # plt.figure(1)
        # plt.subplot(1,3,1)
        # plt.imshow(new_frame[0]+new_mask[0])
        # plt.subplot(1,3,3)
        # plt.imshow(train_frame[0][0].numpy())
        # plt.subplot(1,3,2)
        # plt.imshow(new_mask[0])
        # plt.show()


        train_info_dict = TensorDict({'train_image':train_frame,
                                      'train_anno':label})

        train_info_dict = train_info_dict.apply(stack_tensors)
        return train_info_dict

class Crop3DProcessing():
    def __init__(self,resize_sz,block_mode,scale_patch,train_mode=True,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.resize_sz = resize_sz
        self.mode = block_mode
        self.scale_patch = scale_patch
        self.train_mode = train_mode
        self.shift = True


    def _generate_bbox_label(self, label, bbox_size):
        bbox_list = []
        for key, value in label.items():
            x_coord,y_coord = value['coord'][0],value['coord'][1]
            bbox = [y_coord - bbox_size[0]//2,x_coord - bbox_size[1]//2,y_coord + bbox_size[0]//2,x_coord + bbox_size[1]//2]
            # if self.shift:
            #     bbox = [float(x) for x in bbox]
            #     bbox = prutils.perturb_box(torch.tensor(bbox),min_iou=0.9,sigma_factor=0.1)[0].tolist()
            #     bbox = [int(x) for x in bbox]
            bbox_list.append({'bbox':bbox,'cls':value['class_num']})
        return bbox_list


    def __call__(self,data:TensorDict):
        if self.mode == 'body':
            bbox_size = prutils.generate_bbox_size(data=data,scale=self.scale_patch)
            bbox_list = self._generate_bbox_label(data['body_anno'],bbox_size)
        elif self.mode == 'disc':
            bbox_size = prutils.generate_bbox_size(data=data, scale=self.scale_patch)
            bbox_list = self._generate_bbox_label(data['disc_anno'], bbox_size)
        index = random.sample(range(0,len(bbox_list)),k=3)
        bbox_dict = [bbox_list[idx] for idx in index]
        patch_list,label_list = [],[]
        for bbox_index in bbox_dict:
            x1,y1,x2,y2 = bbox_index['bbox']
            bbox = torch.tensor([x1,y1,x2-x1,y2-y1]).float()
            bbox = prutils.perturb_box(bbox,0.9,sigma_factor=0.1)[0].clamp(min=0).long().tolist()
            class_name = torch.tensor(bbox_index['cls'])
            patch = ((data['train_image'][:,bbox[0]:bbox[0]+bbox[2],bbox[1]:bbox[1]+bbox[3]])/255.0)
            if self.train_mode:
                # random rotate
                angle_list = range(-45,45,1)
                angle = random.sample(angle_list,k=1)[0]
                patch = (rotate_bound(patch.transpose(1, 2, 0),angle)).transpose(2,0,1)

            train_patch = prutils.resize_patch(patch,self.resize_sz)
            # plt.figure(1)
            # for i in range(0,train_patch.shape[0]):
            #     plt.subplot(1,5,i+1)
            #     plt.imshow(train_patch.numpy()[i])
            # plt.savefig('/home/adminer/SPARK/PyDoctor/pydoctor/result_plots/tmp/'+str(time.time())+'.png')
            # plt.show()
            patch_list.append(train_patch)
            label_list.append(class_name)

        train_info_dict = TensorDict({'train_image':patch_list,
                                      'train_anno':label_list})

        train_info_dict = train_info_dict.apply(stack_tensors)
        return train_info_dict












            
        
        
    





