import warnings

import torch.nn as nn
import torch
import collections.abc
import numpy as np
from typing import List, Union
import torch.nn.functional as F
from math import ceil, floor
# from kornia.filters import spatial_gradient3d


def skew(vector):
    """
    skew-symmetric operator for rotation matrix generation
    """

    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def get_rotation_mat(ori1, ori2):
    """
    generating pythonic style rotation matrix
    :param ori1: your current orientation
    :param ori2: orientation to be rotated
    :return: pythonic rotation matrix.
    """
    ori1 = np.array(ori1) if isinstance(ori1, collections.abc.Collection) else ori1
    ori2 = np.array(ori2) if isinstance(ori2, collections.abc.Collection) else ori2
    v = np.cross(ori1, ori2)
    c = np.dot(ori1, ori2)
    mat = np.identity(3) + skew(v) + np.matmul(skew(v), skew(v)) / (1 + c)
    # return torch.from_numpy(mat).float()
    return torch.from_numpy(np.flip(mat).copy()).float().unsqueeze(0).unsqueeze(0)


def get_scaling_mat(scale):
    return torch.diag_embed(torch.tensor(scale)).float()


def cubic_padding(func):
    def wrapper(*args, **kwargs):
        img = args[0]
        b, _, w, h, d = img.shape
        max_dim = max(w, h, d)

        # cubic padding to avoid unreasonably stretching
        padding = [floor((max_dim - d) / 2), ceil((max_dim - d) / 2), floor((max_dim - h) / 2),
                   ceil((max_dim - h) / 2), floor((max_dim - w) / 2), ceil((max_dim - w) / 2)]
        img = F.pad(img, padding)

        return func(*(img, *args[1:]), **kwargs)[:, :, padding[-2]: w + padding[-2], padding[-4]: h + padding[-4],
               padding[-6]: d + padding[-6]]

    return wrapper


@cubic_padding
def affine_transformation(img, affine: Union[List[torch.Tensor], torch.Tensor], pure_translation=None, mode='bilinear'):
    """
    :param img:
    :param affine:
    :param pure_translation:
    :return:
    """
    device = img.device
    b = img.shape[0]

    # add no pure translation
    if pure_translation is None:
        pure_translation = torch.zeros(b, 3, 1).to(device)

    # calculate affine matrices
    affine_mat = torch.eye(3, 3)
    if not isinstance(affine, list):

        affine_mat = affine.squeeze()

    elif len(affine) == 1:

        affine_mat = affine[0]

    else:
        for index in range(len(affine) - 1):
            affine_mat = torch.matmul(affine[index].squeeze().to(device), affine[index + 1].squeeze().to(device))
            affine[index + 1] = affine_mat

    # apply one-step affine transform
    affine_mat = affine_mat.repeat([b, 1, 1]) if len(affine_mat.shape) == 2 else affine_mat
    affine_matrix = torch.cat([affine_mat.to(device), pure_translation], dim=2)
    grid = F.affine_grid(affine_matrix, img.shape, align_corners=False)

    rot_img = F.grid_sample(input=img, grid=grid, mode=mode)
    _, _, w, h, d = rot_img.shape
    return rot_img


def make_coord(shape, flatten=True):
    """
    Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        r = 1 / n
        seq = -1 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def spatial_gradient(img, rank=0.01):

    grad = spatial_gradient3d(img, order=1)

    def gradient_clip(grad_values):

        top_k_rank = grad_values.topk(round(len(grad_values) * rank))

        value_top_k = top_k_rank[0][-1]
        idx_k = top_k_rank[1]

        grad_values[idx_k] = value_top_k

        return grad_values

    pos_idx = grad > 0
    neg_idx = grad < 0
    clipped_pos_grad = gradient_clip(grad[pos_idx])
    clipped_neg_grad = -gradient_clip(-grad[neg_idx])

    grad[pos_idx] = clipped_pos_grad
    grad[neg_idx] = clipped_neg_grad

    return grad


def spatial_flat(img, rank):

    grad = spatial_gradient3d(img, order=1)
    grad = (grad * grad).sum(dim=2)

    seq = grad[grad != 0].flatten()

    top_k_rank = seq.topk(round(len(seq) * rank))

    k_mini = top_k_rank[0][-1]

    mask1 = torch.zeros_like(img)
    mask2 = torch.zeros_like(img)

    mask1[grad < k_mini] = 1
    mask2[grad > 0] = 1

    return mask1 * mask2 * img


def find_bounding_box_3d(volume):
    """
    Find the bounding box of the non-background region in a 3D volume.

    Args:
    volume (torch.Tensor): A 3D tensor representing the volume.
    background_value (int or float): The value of the background voxels.

    Returns:
    tuple: Coordinates of the bounding box (min_x, max_x, min_y, max_y, min_z, max_z).
    """
    # validate the background value

    from scipy.stats import mode
    count = 0

    while count < 10:

        arr = volume.squeeze().flatten().detach().cpu().numpy()
        bg_value = mode(arr)[0]

        if bg_value != 0:

            mask = torch.zeros_like(volume)
            mask[volume != bg_value] = 1
            volume *= mask

            count += 1

        else:

            break

    if 0 < count < 10:
        warnings.warn(f"found background value not 0, the default background value has changed")

    if count == 10:
        warnings.warn("please clear the background before any ongoing reconstruction process!")

    # Identify the foreground voxels
    foreground = volume != 0

    # Find the indices of the foreground voxels
    foreground_positions = foreground.nonzero(as_tuple=True)

    # Find minimum and maximum indices along each dimension
    min_x = torch.min(foreground_positions[-3]) // 2 * 2
    max_x = torch.max(foreground_positions[-3]) // 2 * 2 + 1
    min_y = torch.min(foreground_positions[-2]) // 2 * 2
    max_y = torch.max(foreground_positions[-2]) // 2 * 2 + 1
    min_z = torch.min(foreground_positions[-1]) // 2 * 2
    max_z = torch.max(foreground_positions[-1]) // 2 * 2 + 1

    # even for min, odd for max
    return min_x.item(), max_x.item(), min_y.item(), max_y.item(), min_z.item(), max_z.item()


def crop_volume(volume, target_shape=(192, 192, 144), vox=(1, 1, 1)):
    """
    Crop the volume using the given bounding box.

    Args:
    volume (torch.Tensor): A 3D tensor representing the volume.
    bbox (tuple): Coordinates of the bounding box.

    Returns:
    torch.Tensor: Cropped volume.
    """
    if len(volume.shape) == 3:

        volume = volume.unsqueeze(0).unsqueeze(0)

    bbox = find_bounding_box_3d(volume)
    *_, start_w, end_w = bbox

    if (end_w - start_w) % 2 != 0:

        if start_w > 0:
            start_w -= 1
        elif end_w != volume.shape[-1]:
            end_w += 1
        else:
            raise Exception('The input volume must have an even number of axial slices.')

    target_shape = [192, 192, (end_w - start_w)]

    original_shape = volume.shape[2:]

    start_d = (original_shape[0] - target_shape[0]) // 2
    start_h = (original_shape[1] - target_shape[1]) // 2
    start_w = (original_shape[2] - target_shape[2]) // 2

    # Calculate the end indices
    end_d = start_d + target_shape[0]
    end_h = start_h + target_shape[1]
    end_w = start_w + target_shape[2]

    bbox = start_d, end_d, start_h, end_h, start_w, end_w

    return volume[:, :, start_d: end_d, start_h: end_h, start_w: end_w], bbox


def crop_volume_prev(tensor, voxel_size=(1, 1, 1), target_shape=(192, 192), find_bounding_box=True):
    """
    Crop a 5D tensor into a given shape or by bounding box and remove zero-value slices from the z-dimension.
    The target shape for x-y dimensions will be adjusted based on the voxel size.
    The function returns the cropped tensor and bounding box coordinates.

    Args:
        tensor (torch.Tensor): A 5D tensor with shape [batch, channels, x, y, z].
        voxel_size (tuple): A tuple representing the voxel size in (x, y, z) dimensions. Default is (1, 1, 1).
        target_shape (tuple): A tuple specifying the target (x, y) shape at 1mm isotropic (default is (192, 192)).
        find_bounding_box (bool): If True, the bounding box around non-zero values is used for cropping.
                                  If False, the tensor is cropped or padded to target_shape directly.

    Returns:
        cropped_tensor (torch.Tensor): The cropped tensor with shape [batch, channels, target_x, target_y, z'].
                                       Where z' is the number of non-zero slices in the z dimension.
        bounding_box (dict): A dictionary with start and end points for each dimension:
                             {'x': (start_x, end_x), 'y': (start_y, end_y), 'z': (start_z, end_z)}.

    Raises:
        ValueError: If the calculated bounding box exceeds the proportional target shape.
    """
    assert len(tensor.shape) == 5, "Input tensor must have shape [batch, channels, x, y, z]."
    batch, channels, x_dim, y_dim, z_dim = tensor.shape

    # Calculate the proportional target shape for x and y dimensions based on voxel size
    voxel_x, voxel_y, _ = voxel_size
    proportional_x = int(target_shape[0] / voxel_x)
    proportional_y = int(target_shape[1] / voxel_y)
    proportional_target_shape = (proportional_x, proportional_y)

    # Initialize bounding box coordinates
    start_x, end_x, start_y, end_y, start_z, end_z = 0, x_dim, 0, y_dim, 0, z_dim

    # Step 1: Find bounding box in x-y dimensions if required
    if find_bounding_box:
        # Sum along batch, channels, and z dimensions to get a 2D projection for x-y bounding box detection
        projection = tensor.sum(dim=(0, 1, 4))
        non_zero_indices = torch.nonzero(projection)

        if non_zero_indices.size(0) > 0:
            # Get bounding box coordinates for non-zero values
            min_x, min_y = torch.min(non_zero_indices, dim=0)[0].tolist()
            max_x, max_y = torch.max(non_zero_indices, dim=0)[0].tolist()

            # Check if the bounding box exceeds the proportional target shape
            bounding_box_x = max_x - min_x
            bounding_box_y = max_y - min_y

            if bounding_box_x > proportional_x or bounding_box_y > proportional_y:
                raise ValueError(
                    f"Bounding box dimensions ({bounding_box_x}, {bounding_box_y}) exceed the target shape ({proportional_x}, {proportional_y}).")

            # Calculate the center of the bounding box
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2

            # Calculate new cropping coordinates based on proportional target_shape
            start_x = max(0, center_x - proportional_x // 2)
            start_y = max(0, center_y - proportional_y // 2)

            # Ensure that the cropping coordinates do not exceed the original dimensions
            end_x = min(x_dim, start_x + proportional_x)
            end_y = min(y_dim, start_y + proportional_y)

            # Crop the tensor based on the bounding box coordinates
            tensor = tensor[:, :, start_x:end_x, start_y:end_y, :]
        else:
            # If no non-zero elements are found, return a zero tensor of proportional_target_shape
            tensor = torch.zeros((batch, channels, proportional_x, proportional_y, z_dim), dtype=tensor.dtype,
                                 device=tensor.device)
            # Set bounding box coordinates for an empty tensor
            start_x, end_x, start_y, end_y, start_z, end_z = 0, proportional_x, 0, proportional_y, 0, z_dim
    else:
        # Directly crop to proportional_target_shape from the center of the original image
        center_x, center_y = x_dim // 2, y_dim // 2
        start_x = max(0, center_x - proportional_x // 2)
        start_y = max(0, center_y - proportional_y // 2)

        # Ensure that the cropping coordinates do not exceed the original dimensions
        end_x = min(x_dim, start_x + proportional_x)
        end_y = min(y_dim, start_y + proportional_y)

        # Crop the tensor to the desired shape
        tensor = tensor[:, :, start_x:end_x, start_y:end_y, :]

    # Step 2: Remove zero-value slices in the z-dimension
    z_projection = tensor.sum(dim=(0, 1, 2, 3))  # Project tensor along batch, channels, x, and y dimensions
    non_zero_slices = torch.nonzero(z_projection).squeeze()

    if non_zero_slices.numel() > 0:
        min_z, max_z = non_zero_slices[0].item(), non_zero_slices[-1].item()
        tensor = tensor[:, :, :, :, min_z:max_z + 1]
        start_z, end_z = min_z, max_z + 1
    else:
        # If all slices are zero, keep only one slice with zeros
        tensor = tensor[:, :, :, :, :1]
        start_z, end_z = 0, 1

    # Step 3: Pad if necessary to match the proportional target shape
    _, _, cropped_x, cropped_y, _ = tensor.shape
    pad_x = proportional_target_shape[0] - cropped_x
    pad_y = proportional_target_shape[1] - cropped_y
    if pad_x > 0 or pad_y > 0:
        # Padding format is (z_pad_end, z_pad_start, y_pad_end, y_pad_start, x_pad_end, x_pad_start)
        padding = (0, 0, 0, pad_y, 0, pad_x)  # Pad (z, y, x)
        tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)

    # Create a bounding box dictionary to return
    bounding_box = [start_x, end_x, start_y, end_y, start_z, end_z]

    return tensor, bounding_box


def pad_to_shape(tensor, target_shape):

    assert len(tensor.shape) == 5, "Input tensor must have shape [batch, channels, x, y, z]."
    batch, channels, x_dim, y_dim, z_dim = tensor.shape
    target_x, target_y, target_z = target_shape

    # Calculate the padding needed along each dimension
    pad_x = max(0, target_x - x_dim)
    pad_y = max(0, target_y - y_dim)
    pad_z = max(0, target_z - z_dim)

    # Determine padding for each side
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top
    pad_front = pad_z // 2
    pad_back = pad_z - pad_front

    # Apply padding: format is (pad_back, pad_front, pad_bottom, pad_top, pad_right, pad_left)
    padding = (pad_back, pad_front, pad_bottom, pad_top, pad_right, pad_left)
    padded_tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)

    return padded_tensor


if __name__ == '__main__':

    from handy import *

    img = torch_from_nib_path('H:\\projects\\DIPQSM\\evaluations\\data\\test\\renzo_central_field.nii')
    flat = spatial_flat(img, 0.1)
    save_tensor_as_nii(flat, 'flat')
