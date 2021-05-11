class BaseDoctor:
    """Base class for all doctors."""
    def __init__(self,params):
        self.params = params

    def initialize(self):
        """Overload this function in your doctor. This should initialize the models."""
        raise NotImplementedError

    def diagnose(self,image,index,name):
        raise NotImplementedError

