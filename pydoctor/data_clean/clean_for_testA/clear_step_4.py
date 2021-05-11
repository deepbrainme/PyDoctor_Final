import os
import SimpleITK as sitk
import numpy as np
import pydicom

if __name__ == '__main__':
    # description str need to remove .
    count = 0
    delete_str = ['FST2','320_W','STIR','SAG STIR2D T2W','T2_TIRM_SAG',
                  'FST2_SAGC','T2_STIR(SC)135','T2_TIRM_FS_SAG',
                  'T2_TSE_STIR_SAG_P2']
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
                for str_idx in delete_str:
                    if str_idx not in str_name:
                        continue
                    else:
                        for des_name in des_dict.keys():
                            if str_idx in des_name:
                                to_delete_list = des_dict[des_name]
                                for file in to_delete_list:
                                    if os.path.exists(file):
                                        os.remove(file)

print("Done")

