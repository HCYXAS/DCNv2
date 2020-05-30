#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck
import numpy as np

from dcn_v2 import dcn_v2_conv, DCNv2, DCN
from dcn_v2 import dcn_v2_pooling, DCNv2Pooling, DCNPooling

deformable_groups = 1
N, inC, inH, inW = 2, 2, 4, 4
outC = 2
kH, kW = 3, 3


def conv_identify(weight, bias):
    weight.data.zero_()
    bias.data.zero_()
    o, i, h, w = weight.shape
    y = h//2
    x = w//2
    for p in range(i):
        for q in range(o):
            if p == q:
                weight.data[q, p, y, x] = 1.0


def check_zero_offset():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    conv_mask = nn.Conv2d(inC, deformable_groups * 1 * kH * kW,
                          kernel_size=(kH, kW),
                          stride=(1, 1),
                          padding=(1, 1),
                          bias=True).cuda()

    dcn_v2 = DCNv2(inC, outC, (kH, kW),
                   stride=1, padding=1, dilation=1,
                   deformable_groups=deformable_groups).cuda()

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    conv_mask.weight.data.zero_()
    conv_mask.bias.data.zero_()
    conv_identify(dcn_v2.weight, dcn_v2.bias)

    input_pre=np.array([[[[ 0.2004,  1.7465,  0.5708,  1.1788],
          [-1.3669, -1.4890,  2.9774, -0.3742],
          [ 0.7633, -0.4065,  0.9791,  0.0708],
          [ 1.1339, -1.1434, -0.5540, -0.3194]],

         [[-0.2492,  0.5464,  0.5321,  0.5378],
          [ 0.3070, -0.3816,  0.6656, -1.3091],
          [ 1.2524, -0.4564,  0.4411, -0.2230],
          [ 0.2943, -0.7832, -0.7573,  0.2644]]],


        [[[ 1.2189, -1.2284, -0.0255,  0.6410],
          [-0.2936, -0.7029,  0.2381, -0.6039],
          [-0.8614, -0.9272,  0.2277,  0.1272],
          [ 2.3843,  0.1680, -0.1326,  0.7572]],

         [[ 0.0203, -1.5575,  0.8418,  1.1889],
          [ 0.8645, -1.0511,  2.9121, -0.5233],
          [-0.0840,  1.3571,  0.3699, -1.0240],
          [ 1.1482, -1.1700, -0.7426, -0.2315]]]])

#    input = torch.randn(N, inC, inH, inW).cuda()
    input = torch.from_numpy(input_pre)
    input = input.type(torch.FloatTensor).cuda()
    print(input)

    offset = conv_offset(input)
    mask = conv_mask(input)
    mask = torch.sigmoid(mask)
    output = dcn_v2(input, offset, mask)
    output *= 2
    d = (input - output).abs().max()
    if d < 1e-10:
        print('Zero offset passed')
    else:
        print('Zero offset failed')
        print(input)
        print(output)

def check_gradient_dconv():

    input = torch.rand(N, inC, inH, inW).cuda() * 0.01
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * kW * kH, inH, inW).cuda() * 2
    # offset.data.zero_()
    # offset.data -= 0.5
    offset.requires_grad = True

    mask = torch.rand(N, deformable_groups * 1 * kW * kH, inH, inW).cuda()
    # mask.data.zero_()
    mask.requires_grad = True
    mask = torch.sigmoid(mask)

    weight = torch.randn(outC, inC, kH, kW).cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).cuda()
    bias.requires_grad = True

    stride = 1
    padding = 1
    dilation = 1

    print('check_gradient_dconv: ',
          gradcheck(dcn_v2_conv, (input, offset, mask, weight, bias,
                    stride, padding, dilation, deformable_groups),
                    eps=1e-3, atol=1e-4, rtol=1e-2))


def check_pooling_zero_offset():

    input = torch.randn(2, 16, 64, 64).cuda().zero_()
    input[0, :, 16:26, 16:26] = 1.
    input[1, :, 10:20, 20:30] = 2.
    rois = torch.tensor([
        [0, 65, 65, 103, 103],
        [1, 81, 41, 119, 79],
    ]).cuda().float()
    pooling = DCNv2Pooling(spatial_scale=1.0 / 4,
                           pooled_size=7,
                           output_dim=16,
                           no_trans=True,
                           group_size=1,
                           trans_std=0.0).cuda()

    out = pooling(input, rois, input.new())
    s = ', '.join(['%f' % out[i, :, :, :].mean().item()
                   for i in range(rois.shape[0])])
    print(s)

    dpooling = DCNv2Pooling(spatial_scale=1.0 / 4,
                            pooled_size=7,
                            output_dim=16,
                            no_trans=False,
                            group_size=1,
                            trans_std=0.0).cuda()
    offset = torch.randn(20, 2, 7, 7).cuda().zero_()
    dout = dpooling(input, rois, offset)
    s = ', '.join(['%f' % dout[i, :, :, :].mean().item()
                   for i in range(rois.shape[0])])
    print(s)


def check_gradient_dpooling():
    input = torch.randn(2, 3, 5, 5).cuda() * 0.01
    #input = input.type(torch.DoubleTensor)
    N = 4
    batch_inds = torch.randint(2, (N, 1)).cuda().float()
    x = torch.rand((N, 1)).cuda().float() * 15
    y = torch.rand((N, 1)).cuda().float() * 15
    w = torch.rand((N, 1)).cuda().float() * 10
    h = torch.rand((N, 1)).cuda().float() * 10
    rois = torch.cat((batch_inds, x, y, x + w, y + h), dim=1)
    offset = torch.randn(N, 2, 3, 3).cuda()
    input.requires_grad = True
    offset.requires_grad = True

    spatial_scale = 1.0 / 4
    pooled_size = 3
    output_dim = 3
    no_trans = 0
    group_size = 1
    trans_std = 0.0
    sample_per_part = 4
    part_size = pooled_size

    print('check_gradient_dpooling:',
          gradcheck(dcn_v2_pooling, (input, rois, offset,
                                     spatial_scale,
                                     pooled_size,
                                     output_dim,
                                     no_trans,
                                     group_size,
                                     part_size,
                                     sample_per_part,
                                     trans_std),
                    eps=1e-4))


def example_dconv():
    input = torch.randn(2, 64, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    print("DCN start")
    dcn = DCN(64, 64, kernel_size=(3, 3), stride=1,
              padding=1, deformable_groups=2).cuda()
    #print(dcn.weight.shape, input.shape)
    output = dcn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    print("start backward!!!")
    error.backward()
    print(output.shape)


def example_dpooling():
    input = torch.randn(2, 32, 64, 64).cuda()
    batch_inds = torch.randint(2, (20, 1)).cuda().float()
    x = torch.randint(256, (20, 1)).cuda().float()
    y = torch.randint(256, (20, 1)).cuda().float()
    w = torch.randint(64, (20, 1)).cuda().float()
    h = torch.randint(64, (20, 1)).cuda().float()
    rois = torch.cat((batch_inds, x, y, x + w, y + h), dim=1)
    offset = torch.randn(20, 2, 7, 7).cuda()
    input.requires_grad = True
    offset.requires_grad = True

    # normal roi_align
    pooling = DCNv2Pooling(spatial_scale=1.0 / 4,
                           pooled_size=7,
                           output_dim=32,
                           no_trans=True,
                           group_size=1,
                           trans_std=0.1).cuda()

    # deformable pooling
    dpooling = DCNv2Pooling(spatial_scale=1.0 / 4,
                            pooled_size=7,
                            output_dim=32,
                            no_trans=False,
                            group_size=1,
                            trans_std=0.1).cuda()

    out = pooling(input, rois, offset)
    dout = dpooling(input, rois, offset)
    print(out.shape)
    print(dout.shape)

    target_out = out.new(*out.size())
    target_out.data.uniform_(-0.01, 0.01)
    target_dout = dout.new(*dout.size())
    target_dout.data.uniform_(-0.01, 0.01)
    e = (target_out - out).mean()
    e.backward()
    e = (target_dout - dout).mean()
    e.backward()


def example_mdpooling():
    input = torch.randn(2, 32, 64, 64).cuda()
    input.requires_grad = True
    batch_inds = torch.randint(2, (20, 1)).cuda().float()
    x = torch.randint(256, (20, 1)).cuda().float()
    y = torch.randint(256, (20, 1)).cuda().float()
    w = torch.randint(64, (20, 1)).cuda().float()
    h = torch.randint(64, (20, 1)).cuda().float()
    rois = torch.cat((batch_inds, x, y, x + w, y + h), dim=1)

    # mdformable pooling (V2)
    dpooling = DCNPooling(spatial_scale=1.0 / 4,
                          pooled_size=7,
                          output_dim=32,
                          no_trans=False,
                          group_size=1,
                          trans_std=0.1,
                          deform_fc_dim=1024).cuda()

    dout = dpooling(input, rois)
    target = dout.new(*dout.size())
    target.data.uniform_(-0.1, 0.1)
    error = (target - dout).mean()
    error.backward()
    print(dout.shape)


if __name__ == '__main__':

    print("start!!!")
    example_dconv()
    print("example_dconv!!!")
    example_dpooling()
    example_mdpooling()

    check_pooling_zero_offset()
    # zero offset check
    if inC == outC:
        check_zero_offset()

    check_gradient_dpooling()
    check_gradient_dconv()
    # """
    # ****** Note: backward is not reentrant error may not be a serious problem,
    # ****** since the max error is less than 1e-7,
    # ****** Still looking for what trigger this problem
    # """
