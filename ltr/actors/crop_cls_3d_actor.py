import torch

from . import BaseActor
import matplotlib.pyplot as plt

class Crop3DActor(BaseActor):
    def __init__(self,net,objective,visualize_flag,loss_weight=None):
        super().__init__(net,objective)
        if loss_weight is None:
            loss_weight = {'CrossEntropyLoss':1.0}
        self.loss_weight = loss_weight
        self.visualize = visualize_flag

    """ Actor for training the location """
    def __call__(self, data):
        """
        args:
            data - The input data,

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network
        mask_pred = self.net(data['train_image'])
        anno_label = data['train_anno'].view(-1).long()

        # Compute loss
        loss = self.loss_weight['CrossEntropyLoss'] * self.objective['CrossEntropyLoss'](mask_pred, anno_label)
        pred_index = torch.argmax(mask_pred,dim=-1)
        acc = (pred_index == anno_label).float().sum() / anno_label.size(0)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/CrossEntropyLoss': loss.item(),
                 'acc':acc*100}

        return loss, stats