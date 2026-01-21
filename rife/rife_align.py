# Original Rife Frame Interpolation by hzwer
# https://github.com/megvii-research/ECCV2022-RIFE
# https://github.com/hzwer/Practical-RIFE

# Modifications to use Rife for Image Alignment by tepete ('Enhance Everything!' Discord Server)

# Additional helpful github issues
# https://github.com/megvii-research/ECCV2022-RIFE/issues/278
# https://github.com/megvii-research/ECCV2022-RIFE/issues/344

import sys
import os

import torch
from torch import Tensor
import numpy as np
import cv2
from .IFNet_HDv3_v4_14_align import IFNet
from enum import Enum
import math
from os import path

MAX_VALUES_BY_DTYPE = {
    np.dtype("int8").name: 127,
    np.dtype("uint8").name: 255,
    np.dtype("int16").name: 32767,
    np.dtype("uint16").name: 65535,
    np.dtype("int32").name: 2147483647,
    np.dtype("uint32").name: 4294967295,
    np.dtype("int64").name: 9223372036854775807,
    np.dtype("uint64").name: 18446744073709551615,
    np.dtype("float32").name: 1.0,
    np.dtype("float64").name: 1.0,
}

def get_h_w_c(image: np.ndarray) -> tuple[int, int, int]:
    """Returns the height, width, and number of channels."""
    h, w = image.shape[:2]
    c = 1 if image.ndim == 2 else image.shape[2]
    return h, w, c


def as_3d(img: np.ndarray) -> np.ndarray:
    """Given a grayscale image, this returns an image with 3 dimensions (image.ndim == 3)."""
    if img.ndim == 2:
        return np.expand_dims(img.copy(), axis=2)
    return img

def np_denorm(x: np.ndarray, min_max: tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    """Denormalize from [-1,1] range to [0,1]
    formula: xi' = (xi - mu)/sigma
    Example: "out = (x + 1.0) / 2.0" for denorm
        range (-1,1) to (0,1)
    for use with proper act in Generator output (ie. tanh)
    """
    out = (x - min_max[0]) / (min_max[1] - min_max[0])
    return np.clip(out, 0, 1)

def bgr_to_rgb(image: Tensor) -> Tensor:
    # flip image channels
    # https://github.com/pytorch/pytorch/issues/229
    out: Tensor = image.flip(-3)
    # RGB to BGR #may be faster:
    # out: Tensor = image[[2, 1, 0], :, :]
    return out

def rgb_to_bgr(image: Tensor) -> Tensor:
    # same operation as bgr_to_rgb(), flip image channels
    return bgr_to_rgb(image)

def bgra_to_rgba(image: Tensor) -> Tensor:
    out: Tensor = image[[2, 1, 0, 3], :, :]
    return out


def rgba_to_bgra(image: Tensor) -> Tensor:
    # same operation as bgra_to_rgba(), flip image channels
    return bgra_to_rgba(image)

def norm(x: Tensor):
    """Normalize (z-norm) from [0,1] range to [-1,1]"""
    out = (x - 0.5) * 2.0
    return out.clamp(-1, 1)

def np2tensor(
    img: np.ndarray,
    bgr2rgb: bool = True,
    data_range: float = 1.0,
    normalize: bool = False,
    change_range: bool = True,
    add_batch: bool = True,
) -> Tensor:
    """Converts a numpy image array into a Tensor array.
    Parameters:
        img (numpy array): the input image numpy array
        add_batch (bool): choose if new tensor needs batch dimension added
    """

    # check how many channels the image has, then condition. ie. RGB, RGBA, Gray
    # if bgr2rgb:
    #     img = img[
    #         :, :, [2, 1, 0]
    #     ]  # BGR to RGB -> in numpy, if using OpenCV, else not needed. Only if image has colors.
    if change_range:
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype.name, 1.0)
        t_dtype = np.dtype("float32")
        img = img.astype(t_dtype) / maxval  # ie: uint8 = /255
    # "HWC to CHW" and "numpy to tensor"
    tensor = torch.from_numpy(
        np.ascontiguousarray(np.transpose(as_3d(img), (2, 0, 1)))
    ).float()
    if bgr2rgb:
        # BGR to RGB -> in tensor, if using OpenCV, else not needed. Only if image has colors.)
        if tensor.shape[0] % 3 == 0:
            # RGB or MultixRGB (3xRGB, 5xRGB, etc. For video tensors.)
            tensor = bgr_to_rgb(tensor)
        elif tensor.shape[0] == 4:
            # RGBA
            tensor = bgra_to_rgba(tensor)
    if add_batch:
        # Add fake batch dimension = 1 . squeeze() will remove the dimensions of size 1
        tensor.unsqueeze_(0)
    if normalize:
        tensor = norm(tensor)
    return tensor


def tensor2np(
    img: Tensor,
    rgb2bgr: bool = True,
    remove_batch: bool = True,
    data_range: float = 255,
    denormalize: bool = False,
    change_range: bool = True,
    imtype: type = np.uint8,
) -> np.ndarray:
    """Converts a Tensor array into a numpy image array.
    Parameters:
        img (tensor): the input image tensor array
            4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
        remove_batch (bool): choose if tensor of shape BCHW needs to be squeezed
        denormalize (bool): Used to denormalize from [-1,1] range back to [0,1]
        imtype (type): the desired type of the converted numpy array (np.uint8
            default)
    Output:
        img (np array): 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    n_dim = img.dim()

    # TODO: Check: could denormalize here in tensor form instead, but end result is the same

    img = img.float().cpu()

    img_np: np.ndarray

    if n_dim in (4, 3):
        # if n_dim == 4, has to convert to 3 dimensions
        if n_dim == 4 and remove_batch:
            # remove a fake batch dimension
            img = img.squeeze(dim=0)

        if img.shape[0] == 3 and rgb2bgr:  # RGB
            # RGB to BGR -> in tensor, if using OpenCV, else not needed. Only if image has colors.
            img_np = rgb_to_bgr(img).numpy()
        elif img.shape[0] == 4 and rgb2bgr:  # RGBA
            # RGBA to BGRA -> in tensor, if using OpenCV, else not needed. Only if image has colors.
            img_np = rgba_to_bgra(img).numpy()
        else:
            img_np = img.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC
    elif n_dim == 2:
        img_np = img.numpy()
    else:
        raise TypeError(
            f"Only support 4D, 3D and 2D tensor. But received with dimension: {n_dim:d}"
        )

    # if rgb2bgr:
    # img_np = img_np[[2, 1, 0], :, :] #RGB to BGR -> in numpy, if using OpenCV, else not needed. Only if image has colors.
    # TODO: Check: could denormalize in the begining in tensor form instead
    if denormalize:
        img_np = np_denorm(img_np)  # denormalize if needed
    if change_range:
        img_np = np.clip(
            data_range * img_np, 0, data_range
        ).round()  # np.clip to the data_range

    # has to be in range (0,255) before changing to np.uint8, else np.float32
    return img_np.astype(imtype)


class PrecisionMode(Enum):
    FIFTY_PERCENT = 1
    ONE_HUNDRED_PERCENT = 2
    TWO_HUNDRED_PERCENT = 3
    FOUR_HUNDRED_PERCENT = 4

def calculate_padding(height: int, width: int, modulus: int) -> tuple[int, int]:
    pad_height = (modulus - height % modulus) % modulus
    pad_width = (modulus - width % modulus) % modulus
    return pad_height, pad_width

def rife_align_image(
        clip: np.ndarray,
        ref: np.ndarray,
        precision_mode = PrecisionMode.FIFTY_PERCENT,
        model_path=path.dirname(path.abspath(__file__)),
        model_file='flownet_v4.14.pkl',
        blur=1,
    ) -> np.ndarray:
    device = "cuda"
    fp16 = True
    smooth = 0
    precision = precision_mode.value

    # scales
    s1 = (16, 8, 4, 2)  # needs mod64 pad
    s2 = (8, 4, 2, 1)  # needs mod32 pad
    s3 = (4, 2, 1, 0.5)  # needs mod16 pad
    s4 = (2, 1, 0.5, 0.25)  # needs mod8  pad

    rife_model_path = os.path.join(model_path, model_file)

    state_dict = torch.load(rife_model_path, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    rife_align = IFNet().to(device)
    rife_align.load_state_dict(state_dict, strict=False)
    rife_align.eval()

    if fp16:
        rife_align.half()

    # convert to tensors
    fclip = np2tensor(clip, change_range=True).to(device)
    fref = np2tensor(ref, change_range=True).to(device)
    fmask = None

    # convert to fp16 if needed
    if fp16:
        fclip = fclip.half()
        fref = fref.half()

    with torch.inference_mode():

        # padding for scales
        _, _, fref_h_new, fref_w_new = fref.shape
        if precision in (1, 2):
            p1 = calculate_padding(fref_h_new, fref_w_new, 64)
        if precision > 1:
            p2 = calculate_padding(fref_h_new, fref_w_new, 32)
        if precision > 2:
            p3 = calculate_padding(fref_h_new, fref_w_new, 16)
        if precision > 3:
            p4 = calculate_padding(fref_h_new, fref_w_new, 8)

        # flow based alignment with rife
        with torch.amp.autocast(device, enabled=fp16):  # type: ignore
            if precision == 1:
                fclip = rife_align(
                    fclip,
                    fref,
                    fmask if fmask is not None else None,
                    s1,
                    p1,
                    blur=blur,
                    smooth=smooth * 3 if smooth > 0 else 11,
                    compensate=True,
                    device=device,
                    fp16=fp16,
                )
            elif precision == 2:
                fclip = rife_align(
                    fclip,
                    fref,
                    fmask if fmask is not None else None,
                    s1,
                    p1,
                    blur=9,
                    smooth=91,
                    compensate=False,
                    device=device,
                    fp16=fp16,
                )
                fclip = rife_align(
                    fclip,
                    fref,
                    fmask if fmask is not None else None,
                    s2,
                    p2,
                    blur=blur,
                    smooth=smooth,
                    compensate=True,
                    device=device,
                    fp16=fp16,
                )
            elif precision == 3:
                fclip = rife_align(
                    fclip,
                    fref,
                    fmask if fmask is not None else None,
                    s2,
                    p2,
                    blur=9,
                    smooth=15,
                    compensate=False,
                    device=device,
                    fp16=fp16,
                )
                fclip = rife_align(
                    fclip,
                    fref,
                    fmask if fmask is not None else None,
                    s3,
                    p3,
                    blur=blur,
                    smooth=smooth,
                    compensate=True,
                    device=device,
                    fp16=fp16,
                )
            elif precision == 4:
                fclip = rife_align(
                    fclip,
                    fref,
                    fmask if fmask is not None else None,
                    s2,
                    p2,
                    blur=9,
                    smooth=15,
                    compensate=False,
                    device=device,
                    fp16=fp16,
                )
                fclip = rife_align(
                    fclip,
                    fref,
                    fmask if fmask is not None else None,
                    s3,
                    p3,
                    blur=2,
                    smooth=7,
                    compensate=False,
                    device=device,
                    fp16=fp16,
                )
                fclip = rife_align(
                    fclip,
                    fref,
                    fmask if fmask is not None else None,
                    s4,
                    p4,
                    blur=blur,
                    smooth=smooth,
                    compensate=True,
                    device=device,
                    fp16=fp16,
                )

        return (tensor2np(fclip.squeeze(0).cpu(), change_range=False, imtype=np.float32) * 255).astype(np.uint8)