import math
import random

import numpy as np
import torch
import torch.nn.functional as F

def str_analyse(str_name = None):
    if str_name is None or len(str_name)==2:
        return str_name
    elif len(str_name)<2:
        return 'V1'
    elif len(str_name)>2:
        return str_name.split(',')[0]

def compute_distance_of_two_point(coord_a, coord_b):
    distance = math.sqrt((coord_a[0]-coord_b[0])**2 + (coord_a[1]-coord_b[1])**2)
    return distance

def compute_averate_distance_of_block(data):
    point_list = []
    for key, value in data['body_anno'].items():
        point_list.append(value['coord'] )
    for key, value in data['disc_anno'].items():
        point_list.append(value['coord'] )
    point_list.sort(key=lambda x:x[1], reverse=False)
    sum_distance = 0
    for i, label in enumerate(point_list):
        if i ==0:
            continue
        else:
            distance = compute_distance_of_two_point(point_list[i],point_list[i-1])
            sum_distance +=distance
    average_distance = sum_distance//(len(point_list)-1)
    return average_distance

def generate_guass_kernel(data, scale_factor,sigma_factor):
    average_size_of_body_or_disc = 2*(math.floor(compute_averate_distance_of_block(data))) -1
    kernel_sz = math.floor(scale_factor*average_size_of_body_or_disc)
    sigma = math.floor(sigma_factor*kernel_sz)
    guass_kernel = GaussianTemplate_2D(kernel_size=kernel_sz,sigma=[sigma,sigma])
    return guass_kernel

def generate_bbox_size(data, scale=[1.0,1.0]):
    average_size_of_body_or_disc = 2*(math.floor(compute_averate_distance_of_block(data))) -1
    width = math.floor(scale[0]*average_size_of_body_or_disc)
    high = math.floor(scale[1]*average_size_of_body_or_disc)
    return [width,high]

def GaussianTemplate_2D(kernel_size, sigma=None):
    if sigma is None:
        sigma = [1.0, 1.0]
    template = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    for y in range(kernel_size):
        for x in range(kernel_size):
            y_temp = -(y - center) ** 2 / (2 * sigma[0] * sigma[0])
            x_temp = -(x - center) ** 2 / (2 * sigma[1] * sigma[1])
            template[y][x] = math.exp(y_temp + x_temp) / (math.pow(math.sqrt(2 * math.pi), 3) * sigma[0] * sigma[1])
    template /= template.max()
    return template


def resize_data(frame, label, resize_sz):
    frame = torch.from_numpy(frame).float()
    label = torch.from_numpy(label)
    channel, width,high = frame.shape
    image_out= (F.interpolate(frame.view(1,-1,width,high),(resize_sz,resize_sz),mode='bilinear',align_corners=False))
    label_out= (F.interpolate(label.view(1,1,width,high),(resize_sz,resize_sz),mode='bilinear',align_corners=False))
    return image_out,label_out

def resize_patch(patch, resize_sz):
    patch = torch.from_numpy(patch).float()
    channel, width,high =patch.shape
    patch_out= (F.interpolate(patch.view(1,channel,width,high),(resize_sz,resize_sz),mode='bilinear',align_corners=False))
    return patch_out.squeeze(0)

def rand_uniform(a, b, shape=1):
    """ sample numbers uniformly between a and b.
    args:
        a - lower bound
        b - upper bound
        shape - shape of the output tensor

    returns:
        torch.Tensor - tensor of shape=shape
    """
    return (b - a) * torch.rand(shape) + a

def iou(reference, proposals):
    """Compute the IoU between a reference box with multiple proposal boxes.

    args:
        reference - Tensor of shape (1, 4).
        proposals - Tensor of shape (num_proposals, 4)

    returns:
        torch.Tensor - Tensor of shape (num_proposals,) containing IoU of reference box with each proposal box.
    """

    # Intersection box
    tl = torch.max(reference[:, :2], proposals[:, :2])
    br = torch.min(reference[:, :2] + reference[:, 2:], proposals[:, :2] + proposals[:, 2:])
    sz = (br - tl).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = reference[:, 2:].prod(dim=1) + proposals[:, 2:].prod(dim=1) - intersection

    return intersection / union

def perturb_box(box, min_iou=0.9, sigma_factor=0.1):
    """ Perturb the input box by adding gaussian noise to the co-ordinates

     args:
        box - input box
        min_iou - minimum IoU overlap between input box and the perturbed box
        sigma_factor - amount of perturbation, relative to the box size. Can be either a single element, or a list of
                        sigma_factors, in which case one of them will be uniformly sampled. Further, each of the
                        sigma_factor element can be either a float, or a tensor
                        of shape (4,) specifying the sigma_factor per co-ordinate

    returns:
        torch.Tensor - the perturbed box
    """

    if isinstance(sigma_factor, list):
        # If list, sample one sigma_factor as current sigma factor
        c_sigma_factor = random.choice(sigma_factor)
    else:
        c_sigma_factor = sigma_factor

    if not isinstance(c_sigma_factor, torch.Tensor):
        c_sigma_factor = c_sigma_factor * torch.ones(4)

    perturb_factor = torch.sqrt(box[2] * box[3]) * c_sigma_factor

    # multiple tries to ensure that the perturbed box has iou > min_iou with the input box
    for i_ in range(100):
        c_x = box[0] + 0.5 * box[2]
        c_y = box[1] + 0.5 * box[3]
        c_x_per = random.gauss(c_x, perturb_factor[0])
        c_y_per = random.gauss(c_y, perturb_factor[1])

        w_per = random.gauss(box[2], perturb_factor[2])
        h_per = random.gauss(box[3], perturb_factor[3])

        if w_per <= 1:
            w_per = box[2] * rand_uniform(0.15, 0.5)

        if h_per <= 1:
            h_per = box[3] * rand_uniform(0.15, 0.5)

        box_per = torch.Tensor([c_x_per - 0.5 * w_per, c_y_per - 0.5 * h_per, w_per, h_per]).round()

        if box_per[2] <= 1:
            box_per[2] = box[2] * rand_uniform(0.15, 0.5)

        if box_per[3] <= 1:
            box_per[3] = box[3] * rand_uniform(0.15, 0.5)

        box_iou = iou(box.view(1, 4), box_per.view(1, 4))

        # if there is sufficient overlap, return
        if box_iou > min_iou:
            return box_per, box_iou

        # else reduce the perturb factor
        perturb_factor *= 0.9

    return box_per