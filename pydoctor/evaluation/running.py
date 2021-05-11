import json
import multiprocessing
import os
import time
from itertools import product
from pydoctor.evaluation import Doctor
from pydoctor.evaluation.data import Study
from pydoctor.utils.evalval import evaluate_performace


def save_doctor_result(doctor:Doctor, result:dict,run_id=1):
    """Saves the result of the doctor."""
    if not os.path.exists(doctor.results_dir):
        os.makedirs(doctor.results_dir)
    random_name = time.time()
    json_file = '{}/result_{:03d}_{}.json'.format(doctor.results_dir,run_id,random_name)
    if not os.path.exists(json_file):
        with open(json_file,'w') as file:
            json.dump(result,file)
    else:
        os.remove(json_file)
        with open(json_file,'w') as file:
            json.dump(result,file)
    val = False
    if val:
        groundtruth = '/home/adminer/SPARK/Datasets/lumbar/lumbar_train51_annotation.json'
        evaluate_performace(groundtruth, json_file)





def run_diagnose(std: Study, doctor: Doctor, visdom_info=None,run_id=0):
    """Runs a tracker on a sequence."""

    visdom_info = {} if visdom_info is None else visdom_info
    print('Doctor: {} {},  Study: {}'.format(doctor.name, doctor.parameter_name, std.name))

    try:
        result = doctor.run_study(std, visdom_info=visdom_info)
    except Exception as e:
        print(e)
        return
    return result



def run_dataset(dataset, doctors, threads=0, visdom_info=None,run_id=0,save_flag=True):
    """Runs a list of doctors on a dataset.
    args:
        dataset: List of Study instances, forming a dataset.
        trackers: List of Doctor instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
        visdom_info: Dict containing information about the server for visdom
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(doctors), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    visdom_info = {} if visdom_info is None else visdom_info

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    total_result = []
    if mode == 'sequential':
        for seq in dataset:
            for doctor_info in doctors:
                result = run_diagnose(seq, doctor_info, visdom_info=visdom_info,run_id=run_id)
                total_result.append(result)

    elif mode == 'parallel':
        param_list = [(seq, doctor_info, visdom_info,run_id) for seq, doctor_info in product(dataset, doctors)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_diagnose, param_list)



    if save_flag:
        save_doctor_result(doctor_info,total_result,run_id=run_id)
    print('Done')