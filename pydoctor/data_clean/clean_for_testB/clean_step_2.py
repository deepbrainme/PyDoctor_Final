import json
import SimpleITK as sitk
import pydicom
import os


def read_file(path):
    with open(path, 'r') as f:
        json_file = json.loads(f.read())
    return json_file

def load_anno(anno_path):
    anno_list = read_file(anno_path)
    anno_dict = {}
    for anno in anno_list:
        tmp_dict = {anno['studyUid']: {'seriesUid': anno['seriesUid']}}
        anno_dict.update(tmp_dict)
    return anno_dict


import numpy as np
if __name__ == '__main__':
    # read the dataset and json
    dataset = os.path.abspath('./../../../data/DatasetB/lumbar_testB50')
    jsonpath = os.path.abspath('./../../../data/DatasetB/testB50_series_map.json')
    dirty_data = os.path.abspath('./../../../data/DatasetB/lumbar_testB50/study270/image1.dcm')
    annolist = read_file(jsonpath)
    folder_list = os.listdir(dataset)
    for foldername in folder_list:
        study_path = os.path.join(dataset,foldername)
        filename_list = os.listdir(study_path)
        for filename in filename_list:
            dicom_file_path = os.path.join(study_path,filename)
            dicom_file_info = pydicom.read_file(dicom_file_path)

            # remove the non-sagittal-plane images from given series dicom files
            direction = np.array(dicom_file_info.get(0x00200037).value)
            row_direction = np.array([direction[0], direction[1], direction[2]])
            column_direction = np.array([direction[3], direction[4], direction[5]])
            normal = np.cross(row_direction, column_direction)
            orientation_flag = (-normal[0] > 0.9)

            if orientation_flag :
                continue
            else:
                print('remove non-sagittal')
                if os.path.exists(dicom_file_path):
                    os.remove(dicom_file_path)
        # delete dirty data
        if os.path.exists(dirty_data):
            os.remove(dirty_data)



