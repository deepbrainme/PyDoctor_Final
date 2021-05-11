import os
import SimpleITK as sitk
import numpy as np
import pydicom

if __name__ == '__main__':
    # description str need to remove .
    count = 0
    """Remove the non-T1&T2 images with orientation"""
    dataset_test_path = os.path.abspath('./../../../data/DatasetA/lumbar_testA50')
    folder_list = os.listdir(dataset_test_path)
    for folder_name in folder_list:
        folder_path = os.path.join(dataset_test_path,folder_name)
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder_path)
        str_name = ''
        if len(series_ids)==1:
            count +=1

print("Done")

