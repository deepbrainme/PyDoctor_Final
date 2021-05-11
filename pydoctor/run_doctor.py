import argparse
import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
from pydoctor.evaluation.running import run_dataset
from pydoctor.evaluation import get_dataset
from pydoctor.evaluation import Doctor


def run_doctor(doctor_name, doctor_param, dataset_name='lumbar_test', study_name=None, threads=0, visdom_info=None,run_id=0,save_flag=True):
    """
    Run doctor on study or dataset.
    :param doctor_name: Name of diagnose method
    :param doctor_param: Name of parameter file
    :param dataset: Name of dataset 'lumbar_test' or 'lumbar_val'
    :param study_name: Study name of dataset.
    :param threads: Number of threads.
    :param visdom_info: Dict optionally containing 'use_visdom', 'server' and 'port' for Visdom visualization.
    """
    visdom_info = {} if visdom_info is None else visdom_info
    dataset = get_dataset(dataset_name)

    if study_name is not None:
        for study in dataset:
            if study.name == study_name:
                dataset = [study]

    doctors = [Doctor(doctor_name,doctor_param)]
    run_dataset(dataset,doctors,threads,visdom_info=visdom_info,run_id=run_id,save_flag=save_flag)


def main():
    parser = argparse.ArgumentParser(description='Run doctor on study or dataset.')
    parser.add_argument('--doctor_name',type=str,default='doctorning',help='The name of method in folder parameter')
    parser.add_argument('--doctor_param',type=str,default='version_1',help='The name of param in "parameter/doctor_name/"')
    parser.add_argument('--dataset_name',type=str,default='testB',help='The name of datasets.such as lumbar_testA,lumbarB')
    parser.add_argument('--run_times',type=int,default=10,help='How many times will your run for average.')
    parser.add_argument('--study_name',type=str,default=None ,help='The name of study folder name.')
    parser.add_argument('--threads',type=int, default=0,help='Number of threads.')
    parser.add_argument('--use_visdom',type=bool,default=False,help='Flag to enable visdom')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom')
    parser.add_argument('--runid', type=int, default=666, help='The run id.')
    parser.add_argument('--save_result',type=bool,default=True,help='Flag enable save result.')


    args = parser.parse_args()
    print("Run {} times for average.".format(args.run_times))
    for run_time in range(10):
        print('This is the {} time'.format(run_time+1))
        run_doctor(args.doctor_name, args.doctor_param, args.dataset_name, args.study_name, args.threads,
                    {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port},
                   run_id=args.runid,save_flag=args.save_result)
        print('This is the {} time'.format(run_time+1))
    print("Total Done:Please check the result in PyDoctor/pydoctor/diagnose_result/{}/{}/".format(args.doctor_name,args.doctor_param))



if __name__ == '__main__':
    main()