from pydoctor.features.net_wrappers import NetWithBackbone
from pydoctor.utils import DoctorParams


def parameters():
    params = DoctorParams()

    params.use_gpu = True
    params.image_resize_size = 32 * 16
    params.body_threshold = 0.8
    params.disc_threshold = 0.8
    params.max_order = 6
    params.body_keypoint = 5
    params.disc_keypoint = 6


    params.use_augmentation = True
    params.augmentation = {'fliplr': True,
                           'rotate': [10, -10, 45, -45]
                           }

    # test with our models
    params.body_loc_net = NetWithBackbone(net_path='body_loc_default.pth',use_gpu=params.use_gpu)
    params.disc_loc_net = NetWithBackbone(net_path='disc_loc_default.pth',use_gpu=params.use_gpu)
    params.body_cls_net = NetWithBackbone(net_path='body_cls_resnet50_3.pth',use_gpu=params.use_gpu)
    params.disc_cls_net = NetWithBackbone(net_path='disc_cls_resnet50_3.pth',use_gpu=params.use_gpu)
    # test with you train_restult .set the path like me.
    # params.body_loc_net = NetWithBackbone(net_path='/home/adminer/SPARK/Clean_Code/PyDoctor/workspace/checkpoints/ltr/doctorwang/body_loc_default/ATOMnet_ep0100.pth.tar',use_gpu=params.use_gpu)
    # params.disc_loc_net = NetWithBackbone(net_path='/home/adminer/SPARK/Clean_Code/PyDoctor/workspace/checkpoints/ltr/doctorwang/disc_loc_default/ATOMnet_ep0100.pth.tar',use_gpu=params.use_gpu)
    # params.body_cls_net = NetWithBackbone(net_path='/home/adminer/SPARK/Clean_Code/PyDoctor/workspace/checkpoints/ltr/doctorning/body_cls_resnet50_3/Classnet_ep0004.pth.tar',use_gpu=params.use_gpu)
    # params.disc_cls_net = NetWithBackbone(net_path='/home/adminer/SPARK/Clean_Code/PyDoctor/workspace/checkpoints/ltr/doctorning/disc_cls_resnet50_3/Classnet_ep0004.pth.tar',use_gpu=params.use_gpu)
    return params