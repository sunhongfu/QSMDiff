import math
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import torch.fft as FFT
from functools import partial
import torch.nn.functional as F
from utils.graphic import crop_volume, affine_transformation, get_scaling_mat, crop_volume_prev

from utils.qsm_process import forward_field_calc, truncate_qsm
from utils.myio import *


fft = lambda x: FFT.fftn(x)
ifft = lambda x: FFT.ifftn(x).real


def downsample_by_crop_kspace(img, factor=(1, 1, 1)):
    import torch.fft as fft
    import torch.nn.functional as F

    if all(element == 1 for element in factor):
        return img

    _, _, x, y, z = img.shape
    n_elements = x * y * z

    mask = torch.zeros_like(img)
    mask[img != 0] = 1
    mask = F.interpolate(mask, scale_factor=factor, mode='trilinear')

    x_down, y_down, z_down = int(x * factor[0]), int(y * factor[1]), int(z * factor[2])

    img_freq = fft.fftn(img) / n_elements
    img_shifted_freq = fft.fftshift(img_freq)

    img_crop_shifted_freq = img_shifted_freq[:, :, x // 2 - math.floor(x_down / 2): x // 2 + math.ceil(x_down / 2),
                            y // 2 - math.floor(y_down / 2): y // 2 + math.ceil(y_down / 2),
                            z // 2 - math.floor(z_down / 2): z // 2 + math.ceil(z_down / 2)]

    # img_crop_shifted_freq = img_shifted_freq[:, :, x // 2 - x_down // 2: x // 2 + x_down // 2,
    #                         y // 2 - y_down // 2: y // 2 + y_down // 2,
    #                         z // 2 - z_down // 2: z // 2 + z_down // 2]

    img_crop_freq = fft.ifftshift(img_crop_shifted_freq)

    x_crop = fft.ifftn(img_crop_freq).real * n_elements * (factor[0] * factor[1] * factor[2])

    return x_crop * mask


def upsample(x0, x0_crop, factor, weight=1):

    if all(element == 1 for element in factor):

        return x0_crop

    import torch.fft as fft

    _, _, x, y, z = x0.shape
    _, _, xc, yc, zc = x0_crop.shape

    n_elements = x * y * z
    n_elements_crop = xc * yc * zc

    # mask = torch.zeros_like(x0)
    # mask[x0 != 0] = 1

    x_down, y_down, z_down = int(x * factor[0]), int(y * factor[1]), int(z * factor[2])

    x0_freq = fft.fftn(x0) / n_elements
    x0_crop_freq = fft.fftn(x0_crop) / n_elements_crop

    x0_shifted_freq = fft.fftshift(x0_freq)  # [1, 1, 192, 192, 128] 1mm
    x0_crop_shifted_freq = fft.fftshift(x0_crop_freq)

    free_gen = x0_shifted_freq[:, :, x // 2 - x_down // 2: x // 2 + x_down // 2,
               y // 2 - y_down // 2: y // 2 + y_down // 2,
               z // 2 - z_down // 2: z // 2 + z_down // 2]  # [96, 96, 64]

    # normalize
    # mean = free_gen.mean()
    # mean_crop = x0_crop_shifted_freq.mean()
    # x0_crop_shifted_freq = x0_crop_shifted_freq * mean / mean_crop

    x0_shifted_freq[:, :, x // 2 - x_down // 2: x // 2 + x_down // 2,
    y // 2 - y_down // 2: y // 2 + y_down // 2,
    z // 2 - z_down // 2: z // 2 + z_down // 2] = (1 - weight) * free_gen + weight * x0_crop_shifted_freq

    x0_freq = fft.ifftshift(x0_shifted_freq)

    x0 = fft.ifftn(x0_freq).real * n_elements

    return x0


class TVRegularization(nn.Module):

    def __init__(self, weight, dim=(1, 2, 3)):
        super(TVRegularization, self).__init__()

        if type(weight) is float or type(weight) is int:

            self.weight = weight

        elif isinstance(weight, (list, tuple)):

            self.weights = weight

    def forward(self, x):

        tv_loss = 0

        for i in range(1, 4):

            weight = self.weights[i-1] if self.weights is not None else self.weight

            if i == 1:

                tv_loss += weight * torch.sum(torch.abs(x[:, :, :-1, :, :] - x[:, :, 1:, :, :]))

            elif i == 2:

                tv_loss += weight * torch.sum(torch.abs(x[:, :, :, :-1, :] - x[:, :, :, 1:, :]))

            elif i == 3:

                tv_loss += weight * torch.sum(torch.abs(x[:, :, :, :, :-1] - x[:, :, :, :, 1:]))

        return tv_loss


class DPSGuider:

    def __init__(self, args, img, mask=None, target_shape=(192, 192, 128)):

        self.dim = args.dim

        self.input_vox = args.input_vox  # voxel size of input image.
        self.output_vox = args.output_vox  # expected output image voxel size, should be smaller than 1mm iso.

        self.z_prjs = args.z_prjs
        self.ts_tkd = args.ts_tkd

        if mask is None:
            mask = torch.zeros_like(img)
            mask[img != 0] = 1

        img *= mask

        self.output_shape = [round(d * v) for (d, v) in zip(img.shape[2:], self.input_vox)]  # shape in 1mm iso

        img_1mm_iso = F.interpolate(img, size=self.output_shape)  # input img in 1mm iso

        cropped_img_1mm_iso, bbox = crop_volume(img_1mm_iso)  # cropped 1mm iso
        target_shape = cropped_img_1mm_iso.shape[2:]

        iso_size = img_1mm_iso.shape[2:]

        self.crop_size = [bbox[4], iso_size[2] - bbox[5],
                          bbox[2], iso_size[1] - bbox[3],
                          bbox[0], iso_size[0] - bbox[1]]

        # crop the original image
        org_shape_crop = [round(d / v) for (d, v) in zip(target_shape, self.input_vox)]  # cropped original

        crop_indices = [
            (img.shape[2:][i] - org_shape_crop[i]) // 2 for i in range(len(org_shape_crop))
        ]

        cropped_img = img[...,
                           crop_indices[0]:crop_indices[0] + org_shape_crop[0],
                           crop_indices[1]:crop_indices[1] + org_shape_crop[1],
                           crop_indices[2]:crop_indices[2] + org_shape_crop[2]
                           ]

        # forward calculation if needed
        if args.task == 'DipInv' and args.img_type == 'qsm':

            tmp_mask = torch.zeros_like(cropped_img)
            tmp_mask[cropped_img != 0] = 1
            cropped_measurement = forward_field_calc(cropped_img, vox=self.input_vox, z_prjs=self.z_prjs, need_padding=True) * tmp_mask

        else:

            cropped_measurement = cropped_img

        mask = torch.zeros_like(cropped_measurement)
        mask[cropped_measurement != 0] = 1

        # save lfs and tkd for DipInv guided sampling
        if args.task == 'DipInv':
            self.cropped_lfs = cropped_measurement
            self.cropped_tkd = truncate_qsm(cropped_measurement,
                                                vox=[1, 1, 1], z_prjs=self.z_prjs, ts=self.ts_tkd)[0] * mask

        elif args.task == 'boost':
            self.cropped_lfs = None
            self.cropped_tkd = cropped_img_1mm_iso

        else:
            raise NotImplementedError(args.task + 'is not implemented')

        self.target_shape = target_shape

        self.mask = torch.zeros_like(cropped_img_1mm_iso)
        self.mask[cropped_img_1mm_iso != 0] = 1

        # hyper-params
        self.s1 = args.step_size1  # QSM model step size
        self.s2 = args.step_size2  # QSM model step size
        self.norm = args.norm

        self.lap_weight = args.lap_weight  # laplacian regularization weight
        self.lap_norm = args.lap_norm

        self.tv_weights = args.tv_weights
        self.tv_iterations = args.tv_iterations

        self.ts_tkd = args.ts_tkd

    def crop_and_domain_vox_reverse(self, pred):

        scale = [1/i for i in self.output_vox]

        return downsample_by_crop_kspace(F.pad(pred, self.crop_size), factor=scale)


def tkd_model(img, dipole, ts):

    field_k = fft(img)

    sgn_dipole = torch.sgn(dipole)
    value_dipole = torch.abs(dipole)

    mask_k = value_dipole >= ts
    value_dipole[~mask_k] = ts

    new_dipole = sgn_dipole * value_dipole
    new_dipole[new_dipole == 0] = ts

    measurement = field_k / new_dipole
    measurement[~mask_k] = 0

    return ifft(measurement).real, mask_k


def dc_guide(x0, lfs, dipole, ts=0.2, weight=1, tv=0, dim=3):

    if dim == 2:
        x0 = x0.unsqueeze(0).permute([0, 2, 3, 4, 1])

    measurement, mask_k = tkd_model(lfs, dipole, ts)

    # smooth the measurement
    tv_reg = TVRegularization(tv)
    measurement.requires_grad = True
    measurement = measurement - torch.autograd.grad(tv_reg(measurement), measurement)[0]

    m_k = fft(measurement)
    x0_k = fft(x0)

    x0_k[mask_k == 1] = x0_k[mask_k == 1] * (1 - weight) + m_k[mask_k == 1] * weight
    x0_k[mask_k == 0] = x0_k[mask_k == 0]

    x0 = ifft(x0_k).real

    if dim == 2:
        x0 = x0.squeeze(0).permute([3, 0, 1, 2])

    return x0.real


def dc_guide_on_xt(xt, dipole, noisy_measurement, ts=0.2, weight=1, dim=3):

    if dim == 2:
        xt = xt.unsqueeze(0).permute([0, 2, 3, 4, 1])

    xt_k = fft(xt)
    m_k = fft(noisy_measurement)

    mask_k = torch.zeros_like(dipole)
    mask_k[dipole > ts] = 1

    xt_k[mask_k == 1] = xt_k[mask_k == 1] * (1 - weight) + m_k[mask_k == 1] * weight
    xt_k[mask_k == 0] = xt_k[mask_k == 0]

    x_t = ifft(xt_k).real
    if dim == 2:
        x_t = x_t.squeeze(0).permute([3, 0, 1, 2])

    return x_t