from pydoctor.evaluation.environment import env_settings


class BaseDataset:
    """Base class for all datasets."""
    def __init__(self):
        self.env_settings = env_settings()

    def __len__(self):
        """Overload this function in your dataset. This should return number of sequences in the dataset."""
        raise NotImplementedError

    def get_sequence_list(self):
        """Overload this in your dataset. Should return the list of sequences in the dataset."""
        raise NotImplementedError

class Study:
    """Class for the Study in an evaluation."""
    def __init__(self, name,dataset ,frame_path,index,study_path):
        self.name = name
        self.dataset = dataset
        self.frame_path = frame_path
        self.index = index
        self.study_path =study_path


class StudyList(list):
    """List of sequences. Supports the addition operator to concatenate sequence lists."""
    def __getitem__(self, item):
        if isinstance(item, str):
            for seq in self:
                if seq.name == item:
                    return seq
            raise IndexError('Study name not in the dataset.')
        elif isinstance(item, int):
            return super(StudyList, self).__getitem__(item)
        elif isinstance(item, (tuple, list)):
            return StudyList([super(StudyList, self).__getitem__(i) for i in item])
        else:
            return StudyList(super(StudyList, self).__getitem__(item))

    def __add__(self, other):
        return StudyList(super(StudyList, self).__add__(other))

    def copy(self):
        return StudyList(super(StudyList, self).copy())