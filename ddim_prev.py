"""
Use the deterministic generative process proposed by Song et al. (2020) [1]
[1] Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising Diffusion Implicit Models." International Conference on Learning Representations. 2020.
source file: https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py, Ln 342-356
"""  # noqa
import math
from functools import partial
# from pytorch_msssim import ssim
import torch
from diffusion import GaussianDiffusion, get_beta_schedule
from copy import deepcopy
import torch.nn.functional as F
from utils.guidance import downsample_by_crop_kspace
import torch.fft as fft
from utils.qsm_process import data_fidelity
from torchvision.transforms.functional import crop
from utils.myio import *
from utils.graphic import pad_to_shape


__all__ = ["get_selection_schedule", "DDIM"]



def get_selection_schedule(schedule, size, timesteps):
    """
    :param schedule: selection schedule
    :param size: length of subsequence
    :param timesteps: total timesteps of pretrained ddpm model
    :return: subsequence
    """
    assert schedule in {"linear", "quadratic"}

    if schedule == "linear":
        subsequence = torch.arange(0, timesteps, timesteps // size)
    else:
        subsequence = torch.pow(torch.linspace(0, math.sqrt(timesteps * 0.8), size), 2).round().to(torch.int64)  # noqa

    return subsequence


class DDIM(GaussianDiffusion):
    def __init__(self, betas, model_mean_type, model_var_type, loss_type, eta, subsequence):
        super().__init__(betas, model_mean_type, model_var_type, loss_type)
        self.eta = eta  # coefficient between [0, 1] that decides the behavior of generative process
        self.subsequence = subsequence  # subsequence of the accelerated generation

        eta2 = eta ** 2
        try:
            assert not (eta2 != 1. and model_var_type != "fixed-small"), \
                "Cannot use DDIM (eta < 1) with var type other than `fixed-small`"
        except AssertionError:
            # Automatically convert model_var_type to `fixed-small`
            self.model_var_type = "fixed-small"

        self.alphas_bar = self.alphas_bar[subsequence]
        self.alphas_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64).cuda(), self.alphas_bar[:-1]], dim=0)
        self.alphas = self.alphas_bar / self.alphas_bar_prev
        self.betas = 1. - self.alphas
        self.sqrt_alphas_bar_prev = torch.sqrt(self.alphas_bar_prev)

        # q(x_t|x_0)
        # re-parameterization: x_t(x_0, \epsilon_t)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar)

        self.posterior_var = self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar) * eta2
        self.posterior_logvar_clipped = torch.log(torch.cat([
            self.posterior_var[[1]], self.posterior_var[1:]]).clip(min=1e-20))

        # coefficients to recover x_0 from x_t and \epsilon_t
        self.sqrt_recip_alphas_bar = torch.sqrt(1. / self.alphas_bar)
        self.sqrt_recip_m1_alphas_bar = torch.sqrt(1. / self.alphas_bar - 1.)

        # coefficients to calculate E[x_{t-1}|x_0, x_t]
        self.posterior_mean_coef2 = torch.sqrt(
            1 - self.alphas_bar - eta2 * self.betas
        ) * torch.sqrt(1 - self.alphas_bar_prev) / (1. - self.alphas_bar)
        self.posterior_mean_coef1 = self.sqrt_alphas_bar_prev * \
                                    (1. - torch.sqrt(self.alphas) * self.posterior_mean_coef2)

        # for fixed model_var_type's
        self.fixed_model_var, self.fixed_model_logvar = {
            "fixed-large": (
                self.betas, torch.log(torch.cat([self.posterior_var[[1]], self.betas[1:]]).clip(min=1e-20))),
            "fixed-small": (self.posterior_var, self.posterior_logvar_clipped)
        }[self.model_var_type]

        self.subsequence = torch.as_tensor(subsequence)

    @torch.inference_mode()
    def p_sample(self, denoise_fn, shape, device=torch.device("cpu"), noise=None, seed=None):
        S = len(self.subsequence)
        B, *_ = shape
        subsequence = self.subsequence.to(device)
        _denoise_fn = lambda x, t: denoise_fn(x, subsequence.gather(0, t))
        t = torch.empty((B,), dtype=torch.int64, device=device)
        rng = None
        if seed is not None:
            rng = torch.Generator(device).manual_seed(seed)
        if noise is None:
            x_t = torch.empty(shape, device=device).normal_(generator=rng)
        else:
            x_t = noise.to(device)

        from utils.myio import torch_from_nib_path
        t.fill_(len(subsequence))
        for ti in range(S - 1, -1, -1):
            print(ti)
            t.fill_(ti)
            x_t = self.p_sample_step(_denoise_fn, x_t, t.cuda(), generator=rng, clip_denoised=False)
        # return x_t.reshape([3, 4, 2, 1, 48, 48, 48]).permute([0, 4, 1, 5, 2, 6, 3]).reshape([1, 1, 144, 192, 96])
        return x_t

    def p_sample_3d_guided(self, shape, denoise_fn, guider, device=torch.device("cuda")):

        img_guide = guider.img
        mask = torch.zeros(shape).to(device)
        if guider.factor is not None:
            img_guide = F.interpolate(img_guide, scale_factor=guider.factor)
        mask[img_guide != 0] = 1

        # prepare for diffusion process
        S = len(self.subsequence)
        _, _, h, w, d = shape
        subsequence = self.subsequence.to(device)
        _denoise_fn = lambda x, t: denoise_fn(x, subsequence.gather(0, t))

        x_t = torch.randn(shape, device=device)

        from utils.condition import overlapping_grid_indices
        h_list, w_list, d_list = overlapping_grid_indices(x_t, output_size=48, r=48)
        corners = [(i, j, k) for i in h_list for j in w_list for k in d_list]

        x_grid_mask = torch.zeros_like(x_t)

        for (hi, wj, dk) in corners:
            x_grid_mask[:, :, hi: hi + 48, wj: wj + 48, dk: dk + 48] += 1

        x_t.requires_grad = True

        mini_bs = 24
        bs = len(corners)
        num_mini_batch = bs // mini_bs + 1

        # iterative reverse process starts
        for ti in range(S - 1, -1, -1):

            eps_t = torch.zeros_like(x_t)
            t = torch.empty((1,), dtype=torch.int64, device=device)
            t.fill_(ti)

            x_t_patch = torch.cat([x_t[:, :, hi: hi + 48, wj: wj + 48, dk: dk + 48] for (hi, wj, dk) in corners], dim=0)

            for idx in range(num_mini_batch):

                start = idx * mini_bs
                end = min((idx + 1) * mini_bs, bs)

                with torch.no_grad():
                    out = _denoise_fn(x_t_patch[start: end, :, :, :], t.repeat(end - start))

                for iidx, (hi, wj, dk) in enumerate(corners[start: end]):  # todo: debug here when batch size is 1
                    eps_t[:, :, hi: hi + 48, wj: wj + 48, dk: dk + 48] += out[iidx, :, :, :, :]

            eps_t = torch.div(eps_t, x_grid_mask)

            t = t[0]

            _clip = (lambda x: x.clamp(-1., 1.))
            x_0 = _clip(self._pred_x_0_from_eps(x_t=x_t, eps=eps_t, t=t))
            #
            x_0 = guider.dc_guide(x_0, weight=1) * mask
            # x_0 = guider.tkd_guide(x_0, 0.2, weight=0.5)

            model_mean, *_ = self.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)

            if t == 0:
                noise = torch.zeros_like(x_t)
            else:
                noise = torch.empty_like(x_t).normal_()

            model_logvar = self._extract(self.fixed_model_logvar, t, x_t)
            nonzero_mask = (t > 0).reshape((-1,) + (1,) * (x_t.ndim - 1)).to(x_t)
            x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_logvar) * noise

        return x_t.permute([1, 2, 3, 4, 0])

    def p_guided_3d_slice_sample(self, denoise_fn, guider, device=torch.device("cuda")):

        # prepare for diffusion process
        S = len(self.subsequence)
        subsequence = self.subsequence.to(device)
        _denoise_fn = lambda x, t: denoise_fn(x, subsequence.gather(0, t))

        mask_default = guider.mask
        shape = (1, 1, *guider.target_shape)

        x_t = torch.randn(shape, device=device, requires_grad=True)

        mini_bs = 1

        overlap = 8
        step_size = 16
        x_grid_mask = torch.zeros_like(x_t)

        indices = []

        for idx in range(0, x_t.shape[-1], overlap):

            if idx + step_size > x_t.shape[-1]:

                idx = x_t.shape[-1] - step_size

                indices.append(idx)
                x_grid_mask[:, :, :, :, idx:] += 1
                break

            x_grid_mask[:, :, :, :, idx: idx + step_size] += 1
            indices.append(idx)

        bs = len(indices)

        num_mini_batch = math.ceil(bs / mini_bs)

        # iterative reverse process starts
        for ti in range(S - 1, -1, -1):

            eps_t = torch.zeros_like(x_t)

            t = torch.empty((1,), dtype=torch.int64, device=device)
            t.fill_(ti)

            x_t_patch = torch.cat([x_t[:, :, :, :, idx: idx + step_size] for idx in indices], dim=0)

            for idx in range(num_mini_batch):

                start = idx * mini_bs
                end = min((idx + 1) * mini_bs, bs)

                with torch.no_grad():
                    out = _denoise_fn(x_t_patch[start: end, :, :, :], t.repeat(end - start))

                if guider.dim == 3:

                    for iidx, indice in enumerate(indices[start: end]):  # todo: debug here when batch size is 1
                        eps_t[:, :, :, :, indice: indice + step_size] += out[iidx, :, :, :, :]

            eps_t = torch.div(eps_t, x_grid_mask)

            t = t[0] if guider.dim == 3 else torch.empty((bs,), dtype=torch.int64, device=device).fill_(ti)

            # _clip = (lambda x: x.clamp(-10, 10))
            # x_0 = _clip(self._pred_x_0_from_eps(x_t=x_t, eps=eps_t, t=t))
            x_0 = self._pred_x_0_from_eps(x_t=x_t, eps=eps_t, t=t) * mask_default

            # Guidance on X0 starts here
            from utils.condition import TVRegularization, grad_loss, medi_loss
            from utils.qsm_process import forward_field_calc
            forward = partial(downsample_by_crop_kspace, factor=[1/i for i  in guider.input_vox])

            tv = TVRegularization(guider.tv_weights)

            model_mean, *_ = self.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)

            if torch.any(t == 0):
                noise = torch.zeros_like(x_t)
            else:
                # adds-on
                noise = torch.empty_like(x_t).normal_()

            model_logvar = self._extract(self.fixed_model_logvar, t, x_t)
            nonzero_mask = (t > 0).reshape((-1,) + (1,) * (x_t.ndim - 1)).to(x_t)
            x_t_next = model_mean + nonzero_mask * torch.exp(0.5 * model_logvar) * noise

            if guider.dim == 2:
                x_0 *= mask_default
                x_0 = x_0.permute([1, 2, 3, 0]).unsqueeze(0)

            switch = 1 if ti <= guider.tv_iterations else 0
            dc = forward_field_calc(x_0, guider.z_prjs, guider.input_vox, need_padding=True) * mask_default

            # DPS Starts

            # DPS FineTuning

            # if ti >= 150:
            #     guider.s2 = 1
            # elif 100 <= ti < 150:
            #     guider.s2 = 0.5
            # elif 30 <= ti < 100:
            #     guider.s2 = 0.2
            #     guider.lap_weight = 1
            # elif ti < 30:
            #     guider.s2 = 0
            #     guider.lap_weight = 0

            # if 180 <= ti < 200:
            #     guider.lap_weight = 20
            # elif 150 <= ti < 200:
            #     guider.lap_weight = 10
            # elif 100 <= ti < 150:
            #     guider.lap_weight = 5
            # else:
            #     guider.lap_weight = 2

            if ti <= 20:

                guider.s2 = 0.5
                guider.lap_weight = 0.5

            if ti <= 200:
                x_t_next = x_t_next \
                           - torch.autograd.grad(
                    # guider.s1 * torch.norm(x_0 - guider.cropped_tkd, p=guider.norm) +
                    guider.s2 * torch.norm((forward(dc) - guider.cropped_lfs), p=guider.norm) +
                    # lam * medi_loss(x_0, guider.mag) +
                    # (1-ssim((forward(x_0)+1)/2, (guider.cropped_tkd+1)/2, data_range=1)) * guider.s3 +
                    guider.lap_weight * grad_loss(forward(dc), guider.cropped_lfs,
                                                  order=guider.lap_norm) +
                    switch * tv(x_0), x_t)[0] * mask_default

            # #
            # manifold
            # tv_reg = TVRegularization([4e-5, 4e-5, 4e-5])
            # cropped_tkd = guider.cropped_tkd
            # cropped_tkd.requires_grad = True
            # cropped_tkd = cropped_tkd - torch.autograd.grad(tv_reg(cropped_tkd), cropped_tkd)[0]
            # x_t_next = dc_guide_on_xt(x_t_next, dipole, self.q_sample(cropped_tkd, torch.tensor(ti).cuda()), weight=1,
            #                           dim=guider.dim)

            #
            # QSM to QSM enhancement
            # x_t_next = x_t_next \
            #            - * torch.autograd.grad(guider.s1 * torch.norm(x_0 - guider.cropped_lfs, p=guider.norm) + switch * tv(x_0), x_t)[0] * mask_default

            # Mix of SR & DipInv
            # switch = 1 if ti <= 50 else 0
            # dc = torch.fft.ifftn(dipole * torch.fft.fftn(forward(x_0))).real
            # x_t_next = x_t_next \
            #            - guider.s2 * torch.autograd.grad(guider.s2 * torch.norm(dc - guider.img, p=1) +
            #                                  1 * grad_loss(forward(x_0),
            #                                                  self.q_sample(tkd, torch.tensor(ti).to(device)),
            #                                                  order=2) + switch * tv(x_0), x_t)[0] * mask_default
            # 0.25 * grad_loss(torch.fft.ifftn(dipole * torch.fft.fftn(x_0)).real, guider.img, order=2), x_t)[0] * mask_default

            #
            # SRDipInv
            # x_t_next = x_t_next - 1 * torch.autograd.grad(torch.linalg.norm(torch.fft.ifftn(guider.dipole * torch.fft.fftn(forward(x_0))) - guider.img), x_t)[0]
            # if t <= 10:
            #     from utils.myio import save_tensor_as_nii
            #     save_tensor_as_nii(x_t, 'xt_' + str(t.item()) + '.nii')
            #     save_tensor_as_nii(noisy_measure, 'noisy_measure_' + str(t.item()) + '.nii')
            # x_t_next, _ = condition.conditioning(x_t, x_t_next, x_0, guider.img)

            #
            # single-step test
            from utils.condition import model_chi_to_lap_wphase
            # lap_loss = torch.linalg.norm(model_chi_to_lap_wphase(x_0) - add_on)
            # x_t_next = x_t_next \
            #            - torch.autograd.grad(
            #     # guider.s2 * torch.norm(forward(dc) - guider.cropped_lfs, p=guider.norm) +
            #     # lam * medi_loss(x_0, guider.mag) +
            #     5 * lap_loss
            #     # guider.lap_weight * grad_loss(forward(dc), guider.cropped_lfs, order=guider.lap_norm)
            #     , x_t)[0] * mask_default

            x_t = x_t_next

            print("iteration" + str(ti))

            # save_tensor_as_nii(x_t, 'xt_' + str(ti) + '.nii')
            # save_tensor_as_nii(x_0, 'x0_' + str(ti) + '.nii')
            # save_tensor_as_nii(eps_t, 'eps_' + str(ti) + '.nii')

        return guider.crop_and_domain_vox_reverse(x_0)


    def p_xt_guided_sample(self, denoise_fn, guider, device=torch.device("cuda"), mask_default=None):

        # prepare for diffusion process
        S = len(self.subsequence)
        subsequence = self.subsequence.to(device)
        _denoise_fn = lambda x, t: denoise_fn(x, subsequence.gather(0, t))

        shape = (1, 1, *guider.target_shape)
        mask_default = guider.mask

        mini_bs = 16
        patch_dim = 48

        if guider.dim == 3:

            from utils.condition import overlapping_grid_indices
            from utils.myio import torch_from_nib_path
            x_t = torch.randn(shape, device=device, requires_grad=True)
            # adds-on for 3D
            # x_t = self.q_sample(torch_from_nib_path("G:\projects\DiffusionQSm\evaluations\experiments\compare1\lpcnn_renzo_central_field.nii")
            #                     , torch.tensor(4, dtype=torch.int64, device=device))

            h_list, w_list, d_list = overlapping_grid_indices(x_t, output_size=patch_dim, r=40)  # 40 by default for patch 48
            corners = [(i, j, k) for i in h_list for j in w_list for k in d_list]

            x_grid_mask = torch.zeros_like(x_t)

            for (hi, wj, dk) in corners:
                x_grid_mask[:, :, hi: hi + patch_dim, wj: wj + patch_dim, dk: dk + patch_dim] += 1

            bs = len(corners)

        elif guider.dim == 2:

            bs = mask_default.shape[-1]
            x_t = torch.randn(shape, device=device, requires_grad=True).permute([4, 1, 2, 3, 0]).squeeze(-1)
            # x_t = self.q_sample(guider.cropped_tkd, torch.tensor(5, dtype=torch.int64, device=device)).permute([4, 1, 2, 3, 0]).squeeze(-1)
            # x_t.requires_grad = True
            # # x_t = guider.cropped_lfs.permute([4, 1, 2, 3, 0]).squeeze(-1)
            mask_default = mask_default.permute([4, 1, 2, 3, 0]).squeeze(-1)
            corners = None
            x_grid_mask = torch.ones_like(mask_default)

        else:
            raise NotImplementedError

        num_mini_batch = math.ceil(bs / mini_bs)

        # projection
        # forward = partial(F.interpolate, scale_factor=guider.factor, mode='trilinear')
        # backward = partial(F.interpolate, scale_factor=[1/i for i in guider.factor], mode='trilinear')

        forward = partial(downsample_by_crop_kspace, factor=guider.input_vox)
        from utils.guidance import reverse_crop
        # backward = partial(reverse_crop, x0=torch.zeros_like(x_t), factor=guider.domain_vox, weight=1)
        from utils.guider_prev import dc_guide, dc_guide_on_xt
        # from measurement import SuperResolutionOperator, GaussianNoise
        # from condition_methods import PosteriorSamplingPlus
        # noise = GaussianNoise(sigma=0.05)
        # noise.__name__ = 'gaussian'
        # condition = PosteriorSamplingPlus(SuperResolutionOperator(in_shape=[1, 1, 96, 96, 32], scale_factor=2, device=device), noise)
        from utils.qsm_process import truncate_qsm
        # y = truncate_qsm(guider.img, vox=guider.vox, z_prjs=guider.z_prjs, ts=0.1, padding=False)[0]

        # adds-on keep tune beta_start between [0.01, 0.02]
        weights = get_beta_schedule('linear', 0.0005, 0.003, 200).to(device)
        from utils.qsm_process import generate_dipole
        dipole = generate_dipole(guider.mask.shape, guider.z_prjs, guider.input_vox, device=device)
        # tkd = truncate_qsm(guider.img, vox=guider.vox, z_prjs=guider.z_prjs, ts=0.1, padding=True)[0]

        # iterative reverse process starts
        import time
        start_time = time.time()
        for ti in range(S - 1, -1, -1):
        # for ti in range(3, -1, -1):

            eps_t = torch.zeros_like(x_t)

            t = torch.empty((1,), dtype=torch.int64, device=device)
            t.fill_(ti)

            x_t_patch = torch.cat([x_t[:, :, hi: hi + patch_dim, wj: wj + patch_dim, dk: dk + patch_dim]
                                   for (hi, wj, dk) in corners], dim=0) if guider.dim == 3 else x_t

            for idx in range(num_mini_batch):

                start = idx * mini_bs
                end = min((idx + 1) * mini_bs, bs)

                if start == end:
                    break

                with torch.no_grad():
                    out = _denoise_fn(x_t_patch[start: end, :, :, :], t.repeat(end - start))

                if guider.dim == 3:

                    for iidx, (hi, wj, dk) in enumerate(corners[start: end]):
                        eps_t[:, :, hi: hi + patch_dim, wj: wj + patch_dim, dk: dk + patch_dim] += out[iidx, :, :, :, :]

                else:

                    eps_t[start: end, :, :, :] = out

            eps_t = torch.div(eps_t, x_grid_mask)

            t = t[0] if guider.dim == 3 else torch.empty((bs,), dtype=torch.int64, device=device).fill_(ti)

            _clip = (lambda x: x)
            # _clip = (lambda x: x.clamp(-100., 100.))
            # _clip = (lambda x: x.clamp(-1., 1.))

            x_0 = _clip(self._pred_x_0_from_eps(x_t=x_t, eps=eps_t, t=t)) * mask_default
            # x_0 = (self._pred_x_0_from_eps(x_t=x_t, eps=eps_t, t=t))
            print("iteration" + str(ti) + ': ' + str(x_0.max()))

            # Guidance on X0 starts here
            from utils.condition import TVRegularization, grad_loss
            from utils.qsm_process import forward_field_calc

            tv = TVRegularization(guider.tv_weights)

            # X0 optimization
            # x_0 = x_0.unsqueeze(0).permute([0, 2, 3, 4, 1])
            #
            # for i in range(10):
            #     dc = forward_field_calc(x_0, guider.z_prjs, guider.input_vox, need_padding=True)
            #     x_0 = x_0 - \
            #           guider.s2 * torch.autograd.grad(torch.norm(forward(dc) - guider.cropped_lfs, p=guider.norm) + tv(x_0),
            #                               x_0, retain_graph=True)[0]
            # x_0 = x_0.squeeze(0).permute([3, 0, 1, 2])

            # projection guide
            # x_0 = dc_guide(x_0, guider.cropped_lfs, dipole, weight=0.2, tv=[8e-3, 8e-3, 8e-3], dim=guider.dim,  ts=guider.ts_tkd) * mask_default

            model_mean, *_ = self.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)

            if torch.any(t == 0):
                noise = torch.zeros_like(x_t)
            else:
                # adds-on
                noise = torch.empty_like(x_t).normal_()

            model_logvar = self._extract(self.fixed_model_logvar, t, x_t)
            nonzero_mask = (t > 0).reshape((-1,) + (1,) * (x_t.ndim - 1)).to(x_t)
            x_t_next = model_mean + nonzero_mask * torch.exp(0.5 * model_logvar) * noise

            # # add-on: free generate for last epochs
            # if ti <= 5:
            #     continue

            if guider.dim == 2:

                x_0 *= mask_default
                x_0 = x_0.permute([1, 2, 3, 0]).unsqueeze(0)

            # DipInv
            # switch = 1 if ti <= 200 else 0
            # dc = torch.fft.ifftn(dipole * torch.fft.fftn(forward(x_0))).real
            # x_t_next = x_t_next - guider.s2 * torch.autograd.grad(torch.norm(dc - guider.img, p=1)
            #                                                       + 0.2 * grad_loss(dc, guider.img, order=2) + switch * tv(x_0), x_t)[0]

            switch = 1 if ti <= guider.tv_iterations else 0
            dc = forward_field_calc(x_0, guider.z_prjs, guider.input_vox, need_padding=True)

            from utils.myio import save_tensor_as_nii
            # save_tensor_as_nii(x_t_next * mask_default, 'xt_' + str(ti) + '.nii')

            # DPS
            # for all general cases

            x_t_next = x_t_next \
                       - torch.autograd.grad(guider.s1 * torch.norm(x_0 - guider.cropped_tkd, p=guider.norm) +
                                             guider.s2 * torch.norm(forward(dc) - guider.cropped_lfs, p=guider.norm) +
                                             # (1-ssim((forward(x_0)+1)/2, (guider.cropped_tkd+1)/2, data_range=1)) * guider.s3 +
                                             guider.lap_weight * grad_loss(forward(dc), guider.cropped_lfs,
                                                     order=guider.lap_norm) +
                                             switch * tv(x_0), x_t)[0] * mask_default

            # manifold
            # tv_reg = TVRegularization([4e-5, 4e-5, 4e-5])
            # cropped_tkd = guider.cropped_tkd
            # cropped_tkd.requires_grad = True
            # cropped_tkd = cropped_tkd - torch.autograd.grad(tv_reg(cropped_tkd), cropped_tkd)[0]
            # x_t_next = dc_guide_on_xt(x_t_next, dipole, self.q_sample(cropped_tkd, torch.tensor(ti).cuda()), weight=1, dim=guider.dim)

            # QSM to QSM enhancement

            # x_t_next = x_t_next - \
            #             guider.s2 * torch.autograd.grad(torch.norm(x_0 - guider.cropped_tkd, p=guider.norm)
            #             + guider.lap_weight * grad_loss(forward(x_0), guider.cropped_tkd, order=guider.lap_norm)
            #             + switch * tv(x_0), x_t)[0] * mask_default


            # Mix of SR & DipInv
            # switch = 1 if ti <= 50 else 0
            # dc = torch.fft.ifftn(dipole * torch.fft.fftn(forward(x_0))).real
            # x_t_next = x_t_next \
            #            - guider.s2 * torch.autograd.grad(guider.s2 * torch.norm(dc - guider.img, p=1) +
            #                                  1 * grad_loss(forward(x_0),
            #                                                  self.q_sample(tkd, torch.tensor(ti).to(device)),
            #                                                  order=2) + switch * tv(x_0), x_t)[0] * mask_default
            # 0.25 * grad_loss(torch.fft.ifftn(dipole * torch.fft.fftn(x_0)).real, guider.img, order=2), x_t)[0] * mask_default

            # SRDipInv
            # x_t_next = x_t_next - 1 * torch.autograd.grad(torch.linalg.norm(torch.fft.ifftn(guider.dipole * torch.fft.fftn(forward(x_0))) - guider.img), x_t)[0]

            # if t <= 10:
            #     from utils.myio import save_tensor_as_nii
            #     save_tensor_as_nii(x_t, 'xt_' + str(t.item()) + '.nii')
            #     save_tensor_as_nii(noisy_measure, 'noisy_measure_' + str(t.item()) + '.nii')
            # x_t_next, _ = condition.conditioning(x_t, x_t_next, x_0, guider.img)

            x_t = x_t_next
            save_tensor_as_nii(x_0, 'x0_' + str(ti) + '.nii')
            if ti == 190:
                print(time.time() - start_time)
                pass
        if guider.dim == 2:
            mask_default = mask_default.permute([1, 2, 3, 0])

        return guider.crop_and_domain_vox_reverse(x_0 * mask_default)

    @staticmethod
    def from_ddpm(diffusion, eta, subsequence):
        return DDIM(**{
            k: diffusion.__dict__.get(k, None)
            for k in ["betas", "model_mean_type", "model_var_type", "loss_type"]
        }, eta=eta, subsequence=subsequence)


if __name__ == "__main__":
    subsequence = get_selection_schedule("linear", 10, 1000)
    print(subsequence)
    betas = get_beta_schedule("linear", 0.0001, 0.02, 1000)
    diffusion = GaussianDiffusion(betas, "eps", "fixed-small", "mse")
    print(diffusion.__dict__)
    print(DDIM.from_ddpm(diffusion, eta=0., subsequence=subsequence).__dict__)
