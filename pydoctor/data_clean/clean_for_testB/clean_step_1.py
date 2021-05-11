import json
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


if __name__ == '__main__':
    # read testB data and json for series id.
    dataset = os.path.abspath('./../../../data/DatasetB/lumbar_testB50')
    jsonpath = os.path.abspath('./../../../data/DatasetB/testB50_series_map.json')
    annolist = load_anno(jsonpath)
    folder_list = os.listdir(dataset)

    # read all study and remove the other series .dicom file not in testB50_series_map.json label.
    # We only need diagnose the series which label in json.
    for foldername in folder_list:
        study_path = os.path.join(dataset,foldername)
        filename_list = os.listdir(study_path)
        for filename in filename_list:
            dicom_file_path = os.path.join(study_path,filename)
            dicom_file_info = pydicom.read_file(dicom_file_path)
            study_id = dicom_file_info.get(0x0020000d).value
            series_id = dicom_file_info.get(0x0020000e).value
            anno_series_id = annolist[str(study_id)]['seriesUid']
            if series_id ==anno_series_id:
                continue
            else:
                print('Not it.')
                if os.path.exists(dicom_file_path):
                    os.remove(dicom_file_path)


