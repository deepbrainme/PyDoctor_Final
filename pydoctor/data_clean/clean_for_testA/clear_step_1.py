import os
import SimpleITK as sitk
import numpy as np
import pydicom

if __name__ == '__main__':
    """Remove the non-T1&T2 images with orientation"""
    dataset_test_path = os.path.abspath('./../../../data/DatasetA/lumbar_testA50')
    folder_list = os.listdir(dataset_test_path)
    for folder_name in folder_list:
        folder_path = os.path.join(dataset_test_path,folder_name)
        file_list = os.listdir(folder_path)
        for file_name in  file_list:
            file_path = os.path.join(folder_path,file_name)
            dicom_file = pydicom.read_file(file_path)
            # read the orientation vector for compute the cross product of two (arrays of) vectors.
            direction = np.array(dicom_file.get(0x00200037).value)
            row_direction = np.array([direction[0],direction[1],direction[2]])
            column_direction = np.array([direction[3],direction[4],direction[5]])
            normal = np.cross(row_direction,column_direction)
            orientation_flag = (-normal[0] > 0.9)
            # read the description for this dicom file .
            description = (dicom_file.get(0x0008103e).value).upper()
            description_flag = ('T1' not in description)
            if orientation_flag and description_flag:
                continue
            else:
                if os.path.exists(file_path):
                    os.remove(file_path)

print("Done")

