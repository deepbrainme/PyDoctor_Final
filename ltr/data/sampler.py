import random
import torch.utils.data
from pydoctor import TensorDict


def no_processing(data):
    return data

class DoctorSample(torch.utils.data.Dataset):
    """Class responsible for sampling frames from train dataset studies to
     form batches.Each training sample is a tuple consisting of i) a set of train
      frame,and  ii) a set of annotation """
    def __init__(self, datasets,p_datasets,samples_per_epoch,
                 num_train_frame,processing=no_processing):
        """
        args:
        :param datasets: list of datasets to be used for training.
        :param p_datasets: List containing the probabilities by which each dataset will be sampled.
        :param samples_per_epoch: Number of training samples per epoch
        :param num_train_frame: Number of channel between the key image.
        :param processing: An instance of processing class which performs the necessary processing of the data.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all studies
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]
        self.samples_per_epoch = samples_per_epoch
        self.num_train_frame = num_train_frame
        self.processing = processing

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, item):
        """
        args:
        :param item: int(ignored since we sample randomly)
        :return: TensorDict -dict containing all the dataset blocks.
        """
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        study_id = random.randint(0,dataset.__len__()-1)
        train_frames, body_dict,disc_dict = dataset.get_frames(study_id,self.num_train_frame)
        data = TensorDict({'train_image': train_frames,
                           'body_anno': body_dict,
                           'disc_anno': disc_dict})

        return self.processing(data)

