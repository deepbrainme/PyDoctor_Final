import torch.nn as nn
import ltr.model.backbone as backbones
import ltr.model.head as headmodels
from ltr import model_constructor


class ATOMnet(nn.Module):
    """ ATOM network module"""
    def __init__(self, feature_extractor, kp_regressor, kp_regressor_layer, extractor_grad=True):

        super(ATOMnet, self).__init__()

        self.feature_extractor = feature_extractor
        self.kp_regressor = kp_regressor
        self.kp_regressor_layer = kp_regressor_layer

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))

        train_feat_kpreg = self.get_backbone_kpreg_feat(train_feat)

        # Obtain iou prediction
        iou_pred = self.kp_regressor(train_feat_kpreg)
        return iou_pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.kp_regressor_layer
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)

    def  get_backbone_kpreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.kp_regressor_layer]


class Classnet(nn.Module):
    """ ATOM network module"""
    def __init__(self, feature_extractor, cls_regressor, cls_regressor_layer, extractor_grad=True):

        super(Classnet, self).__init__()

        self.feature_extractor = feature_extractor
        self.cls_regressor = cls_regressor
        self.cls_regressor_layer = cls_regressor_layer

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))

        train_feat_reg = self.get_backbone_reg_feat(train_feat)

        # Obtain iou prediction
        pred = self.cls_regressor(train_feat_reg[0])
        return pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.cls_regressor_layer
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)

    def  get_backbone_reg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.cls_regressor_layer]


class Siamesenet(nn.Module):
    """ ATOM network module"""
    def __init__(self, sag_feature_extractor,ax_feature_extractor, cls_regressor, cls_regressor_layer, extractor_grad=True):

        super(Siamesenet, self).__init__()
        self.sag_feature_extractor = sag_feature_extractor
        self.ax_feature_extractor = ax_feature_extractor
        self.cls_regressor = cls_regressor
        self.cls_regressor_layer = cls_regressor_layer

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs_sag,train_imgs_ax):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """

        # Extract backbone features
        train_backbone_feat_sag = self.extract_sag_backbone_features(train_imgs_sag.reshape(-1, *train_imgs_sag.shape[-3:]))
        train_backbone_feat_ax = self.extract_ax_backbone_features(train_imgs_ax.reshape(-1, *train_imgs_ax.shape[-3:]))

        train_feat_sag = self.get_backbone_feat(train_backbone_feat_sag)
        train_feat_ax = self.get_backbone_feat(train_backbone_feat_ax)

        # Obtain iou prediction
        pred = self.cls_regressor(train_feat_sag,train_feat_ax)
        return pred

    def extract_sag_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.cls_regressor_layer
        return self.sag_feature_extractor(im, layers)

    def extract_ax_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.cls_regressor_layer
        return self.ax_feature_extractor(im, layers)

    def extract_sag_features(self, im, layers):
        return self.sag_feature_extractor(im, layers)

    def extract_ax_features(self, im, layers):
        return self.ax_feature_extractor(im, layers)

    def  get_backbone_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.cls_regressor_layer]

@model_constructor
def atom_resnet18(backbone_pretrained=True,num_cls=2):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    # Bounding box regressor
    predictor = headmodels.Classifier(num_classes=num_cls)

    net = Classnet(feature_extractor=backbone_net, cls_regressor=predictor, cls_regressor_layer=['layer4'],
                  extractor_grad=True)

    return net

@model_constructor
def atom_resnet50_cls(backbone_pretrained=True,num_cls=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # Bounding box regressor
    predictor = headmodels.Classifier_50(num_classes=num_cls)

    net = Classnet(feature_extractor=backbone_net, cls_regressor=predictor, cls_regressor_layer=['layer4'],
                  extractor_grad=True)

    return net


@model_constructor
def atom_resnet50(segm_input_dim=(64, 256, 512, 1024), segm_inter_dim=(4, 16, 32, 64), segm_dim=(64, 64),
                  backbone_pretrained=True):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # Bounding box regressor
    kp_predictor = headmodels.KeyPointNet(segm_input_dim=segm_input_dim, segm_inter_dim=segm_inter_dim,
                                           segm_dim=segm_dim)
    net = ATOMnet(feature_extractor=backbone_net, kp_regressor=kp_predictor, kp_regressor_layer=['conv1', 'layer1','layer2', 'layer3'],
                  extractor_grad=True)

    return net


@model_constructor
def siamese_res18(backbone_pretrained=True,num_cls=2):
    # backbone
    backbone_net_sag = backbones.ournet18(pretrained=backbone_pretrained)
    backbone_net_ax = backbones.ournet18(pretrained=backbone_pretrained)

    # Bounding box regressor
    predictor = headmodels.SiamClassifier(num_classes=num_cls)

    net = Siamesenet(sag_feature_extractor=backbone_net_sag,ax_feature_extractor=backbone_net_ax,
                     cls_regressor=predictor, cls_regressor_layer=['layer4'],extractor_grad=True)

    return net

@model_constructor
def ours_res50(backbone_pretrained=True,num_cls=2):
    # backbone
    backbone_net_sag = backbones.ournet50(pretrained=backbone_pretrained)
    # Bounding box regressor
    predictor = headmodels.Classifier_50(num_classes=num_cls)

    net = Classnet(feature_extractor=backbone_net_sag, cls_regressor=predictor, cls_regressor_layer=['layer4'],
                   extractor_grad=True)

    return net

@model_constructor
def ours_res18(backbone_pretrained=True,num_cls=2):
    # backbone
    backbone_net_sag = backbones.ournet18(pretrained=backbone_pretrained)
    # Bounding box regressor
    predictor = headmodels.Classifier(num_classes=num_cls)

    net = Classnet(feature_extractor=backbone_net_sag, cls_regressor=predictor, cls_regressor_layer=['layer4'],
                   extractor_grad=True)

    return net