import json


def evaluate_performace(predict_json_path, groundtruth_json_path):
    pass

def read_file(path):
    with open(path, 'r') as f:
        json_file = json.loads(f.read())
    return json_file


def load_anno(anno_path):
    anno_list = read_file(anno_path)
    anno_dict = {}
    for anno in anno_list:
        tmp_dict = {anno['studyUid']: {'seriesUid': anno['data'][0]['seriesUid'],
                                       'instanceUid': anno['data'][0]['instanceUid'],
                                       'point': anno['data'][0]['annotation'][0]['data']['point']}}
        anno_dict.update(tmp_dict)
    return anno_dict

if __name__ == '__main__':
    predict_json_path = '/home/adminer/SPARK/Datasets/lumbar/lumbar_train51_annotation.json'
    ground_truth_path =  '/home/adminer/SPARK/Datasets/lumbar/lumbar_train51_annotation.json'

    predict = load_anno(predict_json_path)
    ground_truth = load_anno(predict_json_path)

    print('done')



