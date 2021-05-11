import torch.nn as nn
import torch
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


def conv_no_relu(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes))


class KeyPointNet(nn.Module):
    """ Network module for mask prediction."""
    def __init__(self,segm_input_dim=(64,64,128,256),segm_inter_dim =(4,16,32,64),segm_dim=(64,64),output=1):
        super().__init__()

        self.segment0 = conv(segm_input_dim[3], segm_dim[0], kernel_size=1, padding=0)
        self.segment1 = conv_no_relu(segm_dim[0],segm_dim[1])

        self.s3 = conv(segm_inter_dim[3], segm_inter_dim[2])
        self.s2 = conv(segm_inter_dim[2], segm_inter_dim[2])
        self.s1 = conv(segm_inter_dim[1], segm_inter_dim[1])
        self.s0 = conv(segm_inter_dim[0], segm_inter_dim[0])

        self.f2 = conv(segm_input_dim[2], segm_inter_dim[2])
        self.f1 = conv(segm_input_dim[1], segm_inter_dim[1])
        self.f0 = conv(segm_input_dim[0], segm_inter_dim[0])

        self.post2 = conv(segm_inter_dim[2], segm_inter_dim[1])
        self.post1 = conv(segm_inter_dim[1], segm_inter_dim[0])
        self.post0 = conv_no_relu(segm_inter_dim[0], output)
        self.sigmoid = torch.nn.Sigmoid()

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat_train,*input):
        f_train = self.segment1(self.segment0(feat_train[3]))

        out = self.s3(F.interpolate(f_train, scale_factor=2,mode='bilinear',align_corners=False))
        # batch * 32 * 48 * 48
        out = self.post2(F.interpolate(self.f2(feat_train[2]) + self.s2(out),
                                       scale_factor=2,mode='bilinear',align_corners=False))
        out = self.post1(F.interpolate(self.f1(feat_train[1]) + self.s1(out),
                                       scale_factor=2,mode='bilinear',align_corners=False))
        out = self.post0(F.interpolate(self.f0(feat_train[0]) + self.s0(out),
                                       scale_factor=2,mode='bilinear',align_corners=False))

        return self.sigmoid(out)



# class KeyPointNet(nn.Module):
#     """ Network module for mask prediction."""
#     def __init__(self,segm_input_dim=(64,64,128,256),segm_inter_dim =(4,16,32,64,128),segm_dim=(64,64),output_channel=1):
#         super().__init__()
#
#         self.segment0 = conv(segm_input_dim[3], segm_dim[0], kernel_size=1, padding=0)
#         self.segment1 = conv_no_relu(segm_input_dim[0],segm_input_dim[0])
#
#         self.s3 = conv(segm_inter_dim[3], segm_inter_dim[3])
#         self.s2 = conv(segm_inter_dim[3], segm_inter_dim[3])
#         self.s1 = conv(segm_inter_dim[3], segm_inter_dim[3])
#         self.s0 = conv(segm_inter_dim[3], segm_inter_dim[3])
#
#         self.f2 = conv(segm_input_dim[2], segm_inter_dim[3])
#         self.f1 = conv(segm_input_dim[1], segm_inter_dim[3])
#         self.f0 = conv(segm_input_dim[0], segm_inter_dim[3])
#
#         self.post2 = conv(segm_inter_dim[4], segm_inter_dim[3])
#         self.post1 = conv(segm_inter_dim[4], segm_inter_dim[3])
#         self.post0 = conv(segm_inter_dim[4], segm_inter_dim[3])
#         self.end = conv_no_relu(segm_inter_dim[3], output_channel)
#         self.sigmoid = torch.nn.Sigmoid()
#
#         # Init weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def forward(self, feat_train,*input):
#         f_train = self.segment1(self.segment0(feat_train[3]))
#
#         out = self.s3(F.interpolate(f_train, scale_factor=2,mode='bilinear',align_corners=False))
#         # batch * 32 * 48 * 48
#         out = self.post2(F.interpolate(torch.cat((self.f2(feat_train[2]) , self.s2(out)),dim=1),
#                                        scale_factor=2,mode='bilinear',align_corners=False))
#         out = self.post1(F.interpolate(torch.cat((self.f1(feat_train[1]), self.s1(out)),dim=1),
#                                        scale_factor=2,mode='bilinear',align_corners=False))
#         out = self.post0(F.interpolate(torch.cat((self.f0(feat_train[0]) , self.s0(out)),dim=1),
#                                        scale_factor=2,mode='bilinear',align_corners=False))
#         out = self.sigmoid(self.end(out))
#         return out