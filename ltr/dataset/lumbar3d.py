import json
import os
import SimpleITK as sitk
import numpy
import pydicom
import numpy as np

from ltr.admin.environment import env_settings
from ltr.data.processing_utils import str_analyse
from ltr.dataset.base_dataset import BaseDataset
from pydoctor.evaluation import Study
from pydoctor.evaluation.data import StudyList


def _read_file(path):
    with open(path, 'r') as f:
        json_file = json.loads(f.read())
    return json_file


class Lumbar3d(BaseDataset):
    """
    The Lumbar dataset from official TianChi competition.
    organized as follows.
        -lumbar
            -lumbar_testA50
                -study...
            -lumbar_train150
                -study...
            -lumbar_train51
                -study...
            lumbar_train150_annotation.json
            lumbar_train51_annotation.json
    """

    def __init__(self, root=None, split='train'):
        """
        args:
        :param root:path to the lumbar dataset.
        :param split: string name 'train','val','test'
        """
        root = env_settings().lumbar_dir if root is None else root
        super().__init__('lumbar', root)

        # dataset split for competition.
        if split == 'train':
            self.studies_path = os.path.join(root, 'DatasetA','lumbar_train150')
            self.anno_path = os.path.join(root, 'DatasetA','lumbar_train150_annotation.json')
            self.anno_meta = self._load_anno(self.anno_path)
        elif split == 'val':
            self.studies_path = os.path.join(root, 'DatasetA','lumbar_train51')
            self.anno_path = os.path.join(root, 'DatasetA','lumbar_train51_annotation.json')
            self.anno_meta = self._load_anno(self.anno_path)
        elif split == 'testA':
            self.studies_path = os.path.join(root,'datasetA','lumbar_testA50')
        elif split == 'testB':
            self.studies_path = os.path.join(root, 'datasetB', 'lumbar_testB50')
        else:
            raise ValueError('Unknow split name.')

        # All folders inside the root.
        self.study_list = self._get_study_list()
        self.body_id = {'L1':0,'L2':1,'L3':2,'L4':3,'L5':4}
        self.body_class = {'V1':0,'V2':1}
        self.disc_id = {'T12-L1':0,'L1-L2':1,'L2-L3':2,'L3-L4':3,'L4-L5':4,'L5-S1':5}
        self.disc_class = {'V1':0,'V2':1,'V3':2,'V4':3,'V5':4}

    def _get_study_list(self):
        return os.listdir(self.studies_path)

    def get_name(self):
        return 'lumbar'

    def _get_study_path(self, std_id):
        return os.path.join(self.studies_path, self.study_list[std_id])

    def _get_key_image_info(self, folder,frame_num=3):
        global key_image_path
        reader = sitk.ImageSeriesReader()
        file_path = os.path.join(folder, os.listdir(folder)[0])
        study_uid = pydicom.read_file(file_path).get(0x0020000d).value
        study_meta = self.anno_meta[str(study_uid)]
        dicom_path_list = reader.GetGDCMSeriesFileNames(folder, study_meta['seriesUid'])
        dicom_slice = [[pydicom.read_file(file), file] for file in dicom_path_list]
        dicom_slice.sort(key=lambda x: float(x[0].ImagePositionPatient[0]))
        data_path = dicom_slice[len(dicom_path_list) // 2][1]
        middile_index = study_meta['point'][0]['zIndex']
        frame_list = []
        for dcm_path in range(middile_index - frame_num // 2,middile_index + frame_num // 2 + 1,1):
            frame_list.append(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(dicom_slice[dcm_path][1]))))
        key_image = numpy.stack(frame_list,axis=0)
        key_image = np.uint8((key_image - key_image.min()) / (key_image.max() - key_image.min()) * 255.0)
        return key_image, study_meta['point']


    def _load_anno(self, anno_path):
        anno_list = _read_file(anno_path)
        anno_dict = {}
        for anno in anno_list:
            tmp_dict = {anno['studyUid']: {'seriesUid': anno['data'][0]['seriesUid'],
                                           'instanceUid': anno['data'][0]['instanceUid'],
                                           'point': anno['data'][0]['annotation'][0]['data']['point']}}
            anno_dict.update(tmp_dict)
        return anno_dict

    def _deal_point_dict(self,point_list):
        body_dict,disc_dict = {},{}
        for ann in point_list:
            coord = ann.get('coord',None)
            identification = ann['tag'].get('identification',None)
            if identification in self.body_id:
                class_num = self.body_class[str_analyse(ann['tag'].get('vertebra','v1').upper())]
                body_dict.update({identification:{'coord':coord,'class_num':class_num}})
            elif identification in self.disc_id:
                class_num = self.disc_class[str_analyse(ann['tag'].get('disc','v1').upper())]
                disc_dict.update({identification:{'coord':coord,'class_num':class_num}})
        return body_dict, disc_dict

    def get_frames(self, std_id, frame_num=5,anno=None):
        dicom_folder = self._get_study_path(std_id)
        key_frame,point_list = self._get_key_image_info(dicom_folder)
        body_dict, disc_dict = self._deal_point_dict(point_list)
        return key_frame, body_dict, disc_dict

    def get_study_list(self):
        return StudyList([self._construct_study(s) for s in  self.study_list])

    def _construct_study(self,study_name):
        study_folder_path = os.path.join(self.studies_path,study_name)
        # series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(study_folder_path)
        # for id in series_ids:
        file_list = [os.path.join(study_folder_path,i) for i in os.listdir(study_folder_path)]
        dicom_slice = [[pydicom.read_file(file),file]for file in file_list]
        dicom_slice.sort(key=lambda x:float(x[0].ImagePositionPatient[0]))
        data_path =  dicom_slice[len(file_list)//2][1]

        return Study(name=study_name,dataset='lumbar_test',frame_path=data_path,index=len(file_list)//2)



