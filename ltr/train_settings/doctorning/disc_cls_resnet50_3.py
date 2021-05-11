import torch.nn as nn
from ltr import MultiGPU
from ltr.data import processing, sampler
from ltr.data.loader import LTRLoader
from ltr.dataset.lumbar3d import Lumbar3d
from ltr.trainers import LTRTrainer
import ltr.model.head.atom as atom_models
import ltr.actors as actors
import torch.optim as optim

def run(settings):
    settings.description = "Default train settings for locate the body block of lumbar."
    settings.batch_size =16
    settings.num_workers = 4
    settings.multi_gpu = False
    settings.print_interval = 10
    settings.feature_sz = 4
    settings.resize_sz = settings.feature_sz * 16
    settings.mode = 'disc'
    settings.vis = False
    settings.scale_patch = [0.9,1.5]

    lumbar_train = Lumbar3d(settings.env.lumbar_dir, split='train')
    lumbar_val = Lumbar3d(settings.env.lumbar_dir, split='val')

    data_processing_train = processing.Crop3DProcessing(resize_sz=settings.resize_sz,
                                                      block_mode=settings.mode,
                                                      scale_patch=settings.scale_patch,
                                                      train_mode=True)

    data_processing_val = processing.Crop3DProcessing(resize_sz=settings.resize_sz,
                                                    block_mode=settings.mode,
                                                    scale_patch=settings.scale_patch,
                                                    train_mode=False)
    #
    dataset_train = sampler.DoctorSample([lumbar_train],[1],samples_per_epoch=100*settings.batch_size,processing=data_processing_train,num_train_frame=3)
    dataset_val = sampler.DoctorSample([lumbar_val],[1],samples_per_epoch=50*settings.batch_size,processing=data_processing_val,num_train_frame=3)
    #
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size,
                             num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, epoch_interval=1,stack_dim=0)

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size,
                           num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=1, stack_dim=0)
    #
    net = atom_models.atom_resnet50_cls(backbone_pretrained=True,num_cls=5)
    #
    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    objective = {'CrossEntropyLoss': nn.CrossEntropyLoss()}

    loss_weight = {'CrossEntropyLoss': 1.0}

    actor = actors.Crop3DActor(net=net, objective=objective,visualize_flag= settings.vis, loss_weight=loss_weight)

    # optimizer
    optimizer = optim.Adam([{'params': actor.net.feature_extractor.parameters(), 'lr': 1e-3},
                            {'params': actor.net.cls_regressor.parameters(),'lr':1e-3}],lr=1e-3)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(20, load_latest=True, fail_safe=False)
    print('All is OK.')

