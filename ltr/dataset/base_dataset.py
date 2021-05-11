import torch.utils.data

class BaseDataset(torch.utils.data.Dataset):
    """Base class for datasets."""

    def __init__(self, name, root):
        """
        args:
        :param name: The name of dataset.
        :param root: The root path to the dataset.
        """
        self.name = name
        self.root = root
        self.study_list = []

    def __len__(self):
        """
        Return size of the dataset
        :return:
        """
        return self.get_num_sequence()

    def get_num_sequence(self):
        """
        Number of studies in a dataset.
        :return:
        """
        return len(self.study_list)

    def get_name(self):
        """
        Name of the dataset.
        :return: string - name of the dataset.
        """
        raise NotImplementedError

    def __getitem__(self, item):
        """Not to be used! Check get_frames() instead."""
        return None

    def get_study_info(self,std_id):
        """
        Return information about a particular squences.
        :param std_id: index of the study.
        :return: Dict
        """
        raise NotImplementedError

    def get_frames(self,std_id, frame_ids, anno=None):
        """
        Get a set of frames from a particular study.
        :param std_id:    - index of sequence
        :param frame_ids: -  a list of study file.
        :param anno:  -The annotation for study.
        :return:
        list - List of frames corresponding to key frame.
        dict - A dict containing meta information about the study..
        """
        raise
