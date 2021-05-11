import copy
import os
import time
from collections import Counter
import cv2
import numpy
import torch
import matplotlib.pyplot as plt
import ltr.data.processing_utils as prutils
from ltr.data.processing_utils import compute_distance_of_two_point
from pydoctor.doctors.base import BaseDoctor
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
import pydicom


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x

def compute_distance_list(keypoint_list):
    keypoint_list.sort(key=lambda x:x[1], reverse=True)
    distance_list = []
    for i, label in enumerate(keypoint_list):
        if i ==0:
            continue
        else:
            distance = compute_distance_of_two_point(keypoint_list[i],keypoint_list[i-1])
            distance_list.append(int(distance))
    return distance_list

class DoctorNing(BaseDoctor):

    def initialize_feature(self):
        if not getattr(self, 'features_initialized', False):
            self.params.body_loc_net.initialize()
            self.params.body_cls_net.initialize()
            self.params.disc_loc_net.initialize()
            self.params.disc_cls_net.initialize()
        self.features_initialized = True

    def initialize(self):
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'
        self.initialize_feature()
        self.body_loc_net = self.params.body_loc_net
        self.body_cls_net = self.params.body_cls_net
        self.disc_loc_net = self.params.disc_loc_net
        self.disc_cls_net = self.params.disc_cls_net
        return True

    def diagnose(self, frame_dict, index, name):
        frame = frame_dict['frame']
        frame_cube = frame_dict['cube']
        self.image_sz = torch.tensor([frame.shape[1], frame.shape[2]])
        self.support_sz = self.params.image_resize_size
        self.resized_image, self.resize_factor = self.resize_image(frame, self.support_sz)

        # compute the location stage.
        location_dict = self.get_body_disc_keypoint_dict(frame,name)
        scale_factor = {'body':[1.0,1.3],'disc':[0.9,1.5]}
        # crop the patches from frame with first stage result and put it in dict
        frame_tensor = torch.from_numpy(frame_cube).float()
        patches = self.crop_from_frame(location_dict,frame_tensor,scale_factor=scale_factor)

        with torch.no_grad():
            # extract_backbone features
            body_cls_init_backbone_feat = self.body_cls_net.extract_backbone(patches['body_patches'])
            disc_cls_init_backbone_feat = self.disc_cls_net.extract_backbone(patches['disc_patches'])
            # get the cls feat (FC layer)
            body_cls_backbone_feat = self.body_cls_net.get_backbone_reg_feat(body_cls_init_backbone_feat)
            disc_cls_backbone_feat = self.disc_cls_net.get_backbone_reg_feat(disc_cls_init_backbone_feat)
            # get the class output
            body_cls_pred = self.body_cls_net.cls_regressor(body_cls_backbone_feat[0])
            disc_cls_pred = self.disc_cls_net.cls_regressor(disc_cls_backbone_feat[0])
            # return the predict label.
            body_index_list = torch.argmax(body_cls_pred, dim=-1).view(5,-1).tolist()
            disc_index_list = torch.argmax(disc_cls_pred, dim=-1).view(6,-1).tolist()

        self.body_id = {'4':'L1','3':'L2','2':'L3','1':'L4','0':'L5'}
        self.body_class = {'0':'v1','1':'v2'}
        self.disc_id = {'5':'T12-L1','4':'L1-L2','3':'L2-L3','2':'L3-L4','1':'L4-L5','0':'L5-S1'}
        self.disc_class = {'0':'v1','1':'v2','2':'v3','3':'v4','4':'v5'}
        point = []
        for idx,body_point in enumerate(body_index_list):
            body_cls = self.body_class[str(Counter(body_point).most_common(1)[0][0])]
            coord = location_dict['body']['coord'][idx]
            tag = {'identification':self.body_id[str(idx)],
                   'vertebra':str(body_cls)}
            point_dict = {'tag':tag,'coord':coord,'zIndex':index}
            point.append(point_dict)
        for idx2 ,disc_point in enumerate(disc_index_list):
            disc_cls = self.disc_class[str(Counter(disc_point).most_common(1)[0][0])]
            coord = location_dict['disc']['coord'][idx2]
            tag = {'identification':self.disc_id[str(idx2)],
                   'disc':str(disc_cls)}
            point_dict = {'tag':tag,'coord':coord,'zIndex':index}
            point.append(point_dict)
        annotation = [{'annotator':0,'data':{'point':point}}]
        return annotation


    def resize_image(self, image, resize_sz):
        """
            Resize a pair data ( image and key_point mask_label) into a resize_sz.
            :param image: Src data image
            :param resize_sz: A unified size for a batch.
            :return: return a pair re_sized data.
            """
        image = torch.from_numpy(image).float().unsqueeze(0)
        _, channel, width, high = image.shape
        image_output = (F.interpolate(image, (resize_sz, resize_sz), mode='bilinear',
                                      align_corners=False))
        resize_factor = [width / resize_sz, high / resize_sz]
        return image_output, resize_factor

    def locate_keypoint_from_mask(self, pred_mask, point_num, threshold):
        """
        Locate the key point from the predicted mask.
        :param threshold: the threshold to compute
        :param pred_mask: mask for analyse
        :param max_order: threshold for adjust.
        :param resize_factor: resize factor to restore the coordinate.
        :param total key point:the number key point to located.
        :return:
        """

        keypoint_result, Flag = self.result_analyse_mask(predict=pred_mask, threshold=threshold, keypoint_num=point_num)
        if Flag is not True:
            if len(keypoint_result) < point_num:
                # print('Lost {} point'.format(point_num-len(keypoint_result)))
                #  compute the distance_between_linear_point_list
                distance_list = compute_distance_list(keypoint_result)
                if max(distance_list) > 1.8 * min(distance_list):
                    print('Lost at inner.')

                else:
                    # print('Lost at out')
                    lost_number = point_num - len(keypoint_result)
                    for i in range(0,lost_number):
                        x = keypoint_result[-2][0]-keypoint_result[-1][0]
                        y = keypoint_result[-2][1]-keypoint_result[-1][1]
                        new_point=[keypoint_result[-1][0]-x,keypoint_result[-1][1]-y]
                        keypoint_result.append(new_point)

            if len(keypoint_result)> point_num:
                keypoint_result = keypoint_result[:point_num]

        return keypoint_result, int(np.mean(compute_distance_list(keypoint_result)))

    def result_analyse_mask(self, predict, threshold,  keypoint_num):
        '''
        Analize the mask of network output
        :param predict:  The predict of network output
        :param threshold: the threshold
        :param resize_factor: the list of resize_factor to restore the location
        :return:
        '''
        # get mask
        mask = (predict > predict.max() * threshold).float().cpu().numpy().astype(np.uint8)

        # get contours from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # compute the area for the contours
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) > keypoint_num:
            # remove the little noise with area < 20.
            for idx, area  in enumerate(cnt_area):
                if area <=10:
                    contours.pop(idx)
            # other method need to do .
        location_output = []
        for cont in contours:
            polygon = cont.reshape(-1, 2)
            # with a most suitable circle to fit the polygon of contour
            # return the center and radius
            (loc_x, loc_y), radius = cv2.minEnclosingCircle(polygon)

            location_output.append([int(loc_x), int(loc_y)])
        flag = len(location_output) == keypoint_num
        location_output.sort(key=lambda x: x[1], reverse=True)
        return location_output,flag


    def get_body_disc_keypoint_dict(self,image,name):
        with torch.no_grad():
            # extract_backbone features
            body_loc_init_backbone_feat = self.body_loc_net.extract_backbone(self.resized_image)
            disc_loc_init_backbone_feat = self.disc_loc_net.extract_backbone(self.resized_image)
            # get the reg feat
            body_keypoint_backbone_feat = self.body_loc_net.get_backbone_kpreg_feat(body_loc_init_backbone_feat)
            disc_keypoint_backbone_feat = self.disc_loc_net.get_backbone_kpreg_feat(disc_loc_init_backbone_feat)
            # compute the mask for location
            body_pred = self.body_loc_net.kp_regressor.forward(body_keypoint_backbone_feat)
            disc_pred = self.disc_loc_net.kp_regressor.forward(disc_keypoint_backbone_feat)
        # reshape the pred for processing.
        body_pred = F.interpolate(body_pred, (self.image_sz[0], self.image_sz[1]), mode='bilinear', align_corners=True)
        disc_pred = F.interpolate(disc_pred, (self.image_sz[0], self.image_sz[1]), mode='bilinear', align_corners=True)
        body_pred = body_pred.view(*body_pred.shape[-2:])
        disc_pred = disc_pred.view(*disc_pred.shape[-2:])
        # self.visdom.register(body_pred,'heatmap',2,'body_pred')
        # self.visdom.register(disc_pred,'heatmap',2,'disc_pred')
        body_loc = body_pred.cpu().detach().numpy().astype(np.uint8)
        disc_loc = disc_pred.cpu().detach().numpy().astype(np.uint8)
        if self.visdom is not None:
            self.visdom.register(torch.tensor(body_loc+image[0]), 'heatmap', 2, 'body_pred_heatmap')
            self.visdom.register(torch.tensor(disc_loc+image[0]), 'heatmap', 2, 'disc_pred_heatmap')
            self.visdom.register(torch.tensor(body_loc+image[0]), 'surface', 2, 'body_pred_surface')
            self.visdom.register(torch.tensor(disc_loc+image[0]), 'surface', 2, 'disc_pred_surface')
            self.visdom.register(torch.tensor(body_loc+image[0]), 'contour', 2, 'body_pred_contour')
            self.visdom.register(torch.tensor(disc_loc+image[0]), 'contour', 2, 'disc_pred_contour')

        # for visualize
        # plt.figure(1)
        # plt.subplot(1, 1, 1)
        # plt.imshow(body_loc+image[0])
        # plt.savefig('/home/adminer/SPARK/PyDoctor/pydoctor/result_plots/body_loc/' + name + '.png')
        # plt.show()
        # #
        # plt.figure(1)
        # plt.subplot(1, 1, 1)
        # plt.imshow(disc_loc+image[0])
        # plt.savefig('/home/adminer/SPARK/PyDoctor/pydoctor/result_plots/disc_loc/' + name + '.png')
        # plt.show()

        body_keypoint_list,body_average = self.locate_keypoint_from_mask(body_pred, self.params.body_keypoint,
                                                              self.params.body_threshold)
        disc_keypoint_list,disc_average = self.locate_keypoint_from_mask(disc_pred, self.params.disc_keypoint,
                                                              self.params.disc_threshold)

        return {'body':{'coord':body_keypoint_list,'size':body_average},
                'disc':{'coord':disc_keypoint_list,'size':disc_average}}

    def crop_from_frame(self, location_dict,frame,scale_factor):
        body_coord,body_size = location_dict['body']['coord'],location_dict['body']['size']
        disc_coord,disc_size = location_dict['disc']['coord'],location_dict['disc']['size']
        body_patches = self.generate_bbox_label(body_coord,scale_factor['body'],body_size,11,frame,resize_sz=64)
        disc_patches = self.generate_bbox_label(disc_coord,scale_factor['disc'],disc_size,11,frame,resize_sz=64)
        # self.visdom.register(body_patches,'images',2,'body_patches')
        # self.visdom.register(disc_patches,'images',2,'disc_patches')
        return {'body_patches':body_patches,'disc_patches':disc_patches}

    def generate_bbox_label(self, location_list, scale_factor,size,argument_number,frame,resize_sz=64):
        bbox_size = [scale_factor[0]*size, scale_factor[1]*size]
        bbox_list = []
        for point in location_list:

            x_coord,y_coord = point[1],point[0]
            bbox = torch.tensor([x_coord - bbox_size[0]//2,y_coord - bbox_size[1]//2,bbox_size[0],bbox_size[1]]).round()
            bbox_argue = [bbox]
            patches = []
            for i in range(argument_number-1):
                if True:
                    argued_bbox, iou = prutils.perturb_box(bbox)



                bbox_argue.append(argued_bbox)
            for crop_bbox in bbox_argue:
                x,y,w,h = crop_bbox.tolist()
                x,y,w,h = int(x),int(y),int(w),int(h),
                patch = (F.interpolate((frame[:,x:x+w,y:y+h]).unsqueeze(0),(resize_sz,resize_sz),mode='bilinear',align_corners=False)).squeeze(0)
                patches.append(patch)
            # for idx, pt in enumerate(patches):
            #     self.visdom.register(pt[0],'heatmap',2,str(idx)+'_image')
            bbox_argue = torch.stack(patches,0)
            bbox_list.append(bbox_argue)
        bbox_list = torch.stack(bbox_list,0)
        patches_return = bbox_list.view(-1,patch.shape[0],64,64)
        return  patches_return

    def read_dicom_image(self, study):
        frame_num = 2
        image = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(study.frame_path)))
        key_image = np.expand_dims(np.uint8((image - image.min()) / (image.max() - image.min()) * 255.0), 0).repeat(3,axis=0)

        # get frame_cube
        file_list = [os.path.join(study.study_path, i) for i in os.listdir(study.study_path)]
        dicom_slice = [[pydicom.read_file(file), file] for file in file_list]
        dicom_slice.sort(key=lambda x: float(x[0].ImagePositionPatient[0]))
        middile_index = study.index
        frame_list = []
        for dcm_path in range(middile_index - frame_num//2, middile_index + frame_num//2 + 1, 1):
            frame_list.append(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(dicom_slice[dcm_path][1]))))
        image_cube = numpy.stack(frame_list, axis=0)
        image_cube = np.uint8((image_cube - image_cube.min()) / (image_cube.max() - image_cube.min()) * 255.0)


        pydicom_file = pydicom.read_file(study.frame_path)
        studyUid = pydicom_file.get(0x0020000D)._value
        seriesUid = pydicom_file.get(0x0020000E)._value
        instanceUid = pydicom_file.get(0x00080018)._value
        uid_info = {'studyUid': str(studyUid),
                    'seriesUid': str(seriesUid),
                    'instanceUid': str(instanceUid)}

        image_dict = {'frame':key_image/255.0,'cube':image_cube/255.0}
        return image_dict , uid_info