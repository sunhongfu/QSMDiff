import torch
import torch.nn as nn
import sys
# sys.path.append('G:\projects\DiffusionQSm')
from utils.guider_prev import DPSGuider
from utils.myio import torch_from_nib_path
from diffusion import get_beta_schedule, GaussianDiffusion
from guided_diffusion.unet import UNetModel
from ddim_prev import get_selection_schedule, DDIM


def generate(args):

    model_name = args.model
    use_ddim = args.ddim
    state_dict = torch.load(args.state_dict)

    ddpm_steps = args.ddpm_steps
    ddim_step = args.ddim_steps

    data_path = args.data_path

    device = args.device

    save_name = args.save_name
    # model hyper setting
    #

    if model_name == 'Unet-ADM_3D':
        model = UNetModel(image_size=48, in_channels=1, model_channels=64, out_channels=1, channel_mult=(1, 2, 4, 8),
                      num_res_blocks=3, attention_resolutions=[], dropout=0.2,
                      dims=3).cuda()

    elif model_name == 'Unet-ADM_2D':
        model = UNetModel(image_size=192, in_channels=1, model_channels=64, out_channels=1, channel_mult=(1, 2, 4, 8),
                      num_res_blocks=3, attention_resolutions=[], dropout=0.2,
                      dims=2).cuda()
        #
        # model = UNetModel(image_size=192, in_channels=1, model_channels=128, out_channels=1, channel_mult=(1, 1, 2, 4),
        #               num_res_blocks=2, attention_resolutions=[48, 24], dropout=0.2, num_head_channels=64,
        #               dims=2).cuda()  # wiener

    args.dim = int(model_name[-2])

    img = torch_from_nib_path(data_path).squeeze(-1) / 0.15

    mask = torch.zeros_like(img)
    mask[img != 0] = 1

    guider = DPSGuider(args, img, mask,  target_shape=None)  # specify target_shape otherwise calculating from vox

    device = torch.device(device)

    model = nn.DataParallel(model.to(device))
    model.load_state_dict(state_dict['model'])
    # model.load_state_dict(state_dict)

    betas = get_beta_schedule('linear', 0.0001, 0.02, ddpm_steps).to(device)

    if use_ddim:

        subsequence = get_selection_schedule('linear', size=ddim_step, timesteps=ddpm_steps)
        diffusion = DDIM(betas, model_mean_type='eps', model_var_type='fixed-small', loss_type='mse', eta=0.5,
                         subsequence=subsequence)

    else:
        diffusion = GaussianDiffusion(betas, model_mean_type='eps', model_var_type='fixed-small', loss_type='mse')

    model.eval()

    pred = diffusion.p_guided_3d_slice_sample(model, device=device, guider=guider) * 0.15
    # pred = diffusion.p_xt_guided_sample(model, device=device, guider=guider, mask_default=mask) * 0.15
    # pred = diffusion.p_sample(
    #     denoise_fn=model, shape=[6, 1, 192, 192, 16],
    #     device=device)    # save_name = '-'.join([model_name[-2:], task, ''.join(map(str, vox)), ''.join(map(str, z_prjs)),

    from utils.myio import save_tensor_as_nii
    save_tensor_as_nii(pred, save_name + '.nii')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--state_dict', type=str, default='G:\projects\DiffusionQSm\\new_episodes\ep9\\chkpt.pkl') # SOTA 3D chunk *

    parser.add_argument('--model', type=str, default='Unet-ADM_3D', choices=['Unet-ADM_3D', 'Unet-ADM_2D'])
    parser.add_argument('--ddim', type=bool, default=True)

    parser.add_argument('--ddpm_steps', type=int, default=1000)
    parser.add_argument('--ddim_steps', type=int, default=200)

    parser.add_argument('--data_path', type=str, default="C:\\Users\\trace\Documents\WeChat Files\s673242975\FileStorage\File\\2025-02\lfs_resharp001.nii")  # noise level 0.01

    # deprecated
    # parser.add_argument('--mag_path', type=str, default="G:\projects\DiffusionQSm\evaluations\data\cosmos_2_cropped.nii")  # noise level 0.01

    parser.add_argument('--input_vox', type=float, nargs=3, default=[1.5, 1.5, 1.5])  # input image vox
    parser.add_argument('--output_vox', type=float, nargs=3, default=[1.5, 1.5, 1.5])  # output image vox
    parser.add_argument('--z_prjs', type=float, nargs=3, default=[0, 0, 1])

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_name', type=str, default='pred')

    parser.add_argument('--task', type=str, default='DipInv', choices=['DipInv', 'boost'])
    parser.add_argument('-it', '--img_type', type=str, default='lfs', choices=['lfs', 'qsm'])

    # Hyper-params
    parser.add_argument('--ts_tkd', type=float, default=0.2, help='threshold for DipInv only')
    parser.add_argument('--tv_weights', type=float, nargs=3, default=[5e-5, 5e-5, 5e-5], help='weight for 3D tv loss')  # 1e-4, 1e-4, 1e-3 for 2D, #3e-4, 3e-4, 3e-4 for 3D
    parser.add_argument('-tv', '--tv_iterations', type=float, default=200)
    parser.add_argument('-s1', '--step_size1', type=float, default=0, help='tkd step_size for DPS') # 2.5 for 2D SR,  0.002 for L1 # 2.5 for 3D by default
    parser.add_argument('-s2', '--step_size2', type=float, default=2, help='lfs step_size for DPS') # 2.5 for 2D SR,  0.002 for L1 # 2.5 for 3D by default
    parser.add_argument('-lw', '--lap_weight', type=float, default=4, help='step_size for DPS') # 2.5 for 2D SR,  0.002 for L1  # 0.2 for tkd lap
    parser.add_argument('-ln', '--lap_norm', type=float, default=2, help='step_size for DPS') # 2.5 for 2D SR,  0.002 for L1
    parser.add_argument('-n', '--norm', type=int, default=2)

    args = parser.parse_args()

    generate(args)
