import os
import SimpleITK as sitk
import numpy as np
import pydicom

if __name__ == '__main__':
    """Remove the num < 6 series sequence."""
    dataset_test_path = os.path.abspath('./../../../data/DatasetA/lumbar_testA50')
    folder_list = os.listdir(dataset_test_path)
    for folder_name in folder_list:
        folder_path = os.path.join(dataset_test_path,folder_name)
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder_path)
        for ids in series_ids:
            file_list = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder_path,ids)
            if len(file_list) < 6:
                for file_path in  file_list:
                    if os.path.exists(file_path):
                        os.remove(file_path)
print("Done")

