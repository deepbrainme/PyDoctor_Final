from ltr.dataset import Lumbar
import torchvision
import torch.nn as nn
from ltr import MultiGPU
from ltr.data import processing, sampler
from ltr.data.loader import LTRLoader
from ltr.dataset.lumbar import Lumbar
from ltr.trainers import LTRTrainer
import ltr.model.head.atom as atom_models
import ltr.actors as actors
import torch.optim as optim

def run(settings):
    settings.description = "Default train settings for locate the body block of lumbar."
    settings.batch_size = 10
    settings.num_workers = 4
    settings.multi_gpu = False
    settings.print_interval = 10
    settings.feature_sz = 32
    settings.resize_sz = settings.feature_sz * 16
    settings.mode = 'body'
    settings.vis = False
    settings.guass_params = {'scale':0.3,'sigma':1.0}
    settings.argument_params = {'rotate_angle':[-45,45]}

    lumbar_train = Lumbar(settings.env.lumbar_dir, split='train')
    lumbar_val = Lumbar(settings.env.lumbar_dir, split='val')

    data_processing_train = processing.LocProcessing(resize_sz=settings.resize_sz,
                                                     block_mode=settings.mode,
                                                     train_mode=True,
                                                     guass_params=settings.guass_params,
                                                     arguement_params=settings.argument_params)

    data_processing_val = processing.LocProcessing(resize_sz=settings.resize_sz,
                                                   block_mode=settings.mode,
                                                   train_mode=False,
                                                   guass_params=settings.guass_params,
                                                   arguement_params=settings.argument_params)

    dataset_train = sampler.DoctorSample([lumbar_train],[1],samples_per_epoch=500*settings.batch_size,processing=data_processing_train,num_train_frame=3)
    dataset_val = sampler.DoctorSample([lumbar_val],[1],samples_per_epoch=50*settings.batch_size,processing=data_processing_val,num_train_frame=3)

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size,
                             num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size,
                           num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)

    net = atom_models.atom_resnet50(backbone_pretrained=True)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    objective = {'mse': nn.MSELoss()}

    loss_weight = {'mse': 1.0}

    actor = actors.LocActor(net=net, objective=objective,visualize_flag= settings.vis, loss_weight=loss_weight)

    # optimizer
    optimizer = optim.Adam([{'params': actor.net.feature_extractor.parameters(), 'lr': 1e-3},
                            {'params': actor.net.kp_regressor.parameters(), 'lr': 1e-3}], lr=1e-3)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(100, load_latest=True, fail_safe=True)



