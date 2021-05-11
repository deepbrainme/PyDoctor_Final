import os
import SimpleITK as sitk
import numpy as np
import pydicom

if __name__ == '__main__':
    count = 0
    """Remove the non-T1&T2 images with orientation"""
    dataset_test_path = os.path.abspath('./../../../data/DatasetA/lumbar_testA50')
    folder_list = os.listdir(dataset_test_path)
    for folder_name in folder_list:
        folder_path = os.path.join(dataset_test_path,folder_name)
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder_path)
        str_name = ''
        if len(series_ids)>1:
            count+=1
            des_dict = {}
            for srs_id in series_ids:
                file_list = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder_path,srs_id)
                dicom_file = pydicom.read_file(file_list[0])
                # read the description
                description = (dicom_file.get(0x0008103e).value).upper()
                str_name +=description
                tmp_dict = {description:list(file_list)}
                des_dict.update(tmp_dict)
            if 'T2' not in str_name:
                continue
            else:
                for des_name in des_dict.keys():
                    if 'T2' not in des_name:
                        to_delete_list = des_dict[des_name]
                        for file in to_delete_list:
                            if os.path.exists(file):
                                os.remove(file)

print("Done")

