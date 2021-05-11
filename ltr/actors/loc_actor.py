from . import BaseActor
import matplotlib.pyplot as plt

class LocActor(BaseActor):
    def __init__(self,net,objective,visualize_flag,loss_weight=None):
        super().__init__(net,objective)
        if loss_weight is None:
            loss_weight = {'mse':1.0}
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

        mask_pred = mask_pred.view(-1,mask_pred.shape[-2],mask_pred.shape[-1])
        anno_label = data['train_anno'].view(-1,mask_pred.shape[-2],mask_pred.shape[-1])
        if self.visualize:
            plt.figure(1)
            plt.subplot(1,3,1)
            plt.imshow(mask_pred[0].cpu().detach().numpy())
            plt.subplot(1, 3, 2)
            plt.imshow(anno_label[0].cpu().detach().numpy())
            plt.subplot(1, 3, 3)
            plt.imshow(data['train_image'][0][0][0].cpu().detach().numpy()+anno_label[0].cpu().detach().numpy())
            plt.show()

        # Compute loss
        loss = self.loss_weight['mse'] * self.objective['mse'](mask_pred, anno_label)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/mse': loss.item()}

        return loss, stats