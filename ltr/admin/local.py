import os


class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = os.path.abspath('./../workspace')   # Base directory for savinng network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'   # Directory for tensorboard files.
        self.lumbar_dir = os.path.abspath('./../data' )  # Directory for lumbar dataset from TianChi Spine.
