"""
Produces objects and masks
"""

from typing import Tuple
from Param import Param
from PIL import Image
from utils import rxy_map, circle, rectangle, lp_filt
import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.fft import fftn, ifftn, fftshift
except ImportError as error:
    print("No cupy, using numpy instead")
    import numpy as cp
    from scipy.fft import fftn, ifftn, fftshift


def load_masks(p: Param) -> Tuple[cp.ndarray, cp.ndarray]:
    r_grid, x_grid, _ = rxy_map(p.scenario.camera_n)
    use_filter = True  # Valid for all except 'unfiltered_phase'
    supports = []
    for shape, size in zip(
        [p.mask.supp_out_shape, p.mask.supp_shape],
        [p.mask.supp_out_size, p.mask.supp_size],
    ):
        match shape:
            case "square":
                supports.append(
                    lp_filt(
                        rectangle(p.mask.camera_n, (size,) * 2),
                        "gaussian",
                        2,
                    )
                )
            case "circ":
                supports.append(
                    lp_filt(circle(p.mask.camera_n, size / 2), "gaussian", 2)
                )
            case _:
                raise TypeError('supp_shape should be "square" or "circ"')
    supp_out, supp = supports

    masks = cp.zeros(
        (p.scenario.camera_n, p.scenario.camera_n, p.scenario.mask_num), dtype=complex
    )
    for k in range(p.scenario.mask_num):
        speckles = (
            (
                cp.random.rand(p.scenario.camera_n, p.scenario.camera_n)
                + 1.0j * cp.random.rand(p.scenario.camera_n, p.scenario.camera_n)
            )
            * 2
            - (1 + 1.0j)
        ) / 2
        match p.mask.mask_type:
            case "speckles":
                mask = speckles * supp
            case "phase":
                speckles = lp_filt(cp.real(speckles), "gaussian", 15)
                mask = cp.exp(1j * speckles * cp.pi * 1e3) * supp
                for ii in range(20):
                    mask = cp.exp(1.0j * cp.angle(mask)) * supp
                    mask = lp_filt(
                        mask,
                        p.mask.filter_shape,
                        p.mask.filter_outer_limit,
                    )
                    mask = mask - lp_filt(
                        mask,
                        p.mask.filter_shape,
                        p.mask.filter_inner_limit,
                    )
            case "unfiltered_phase":
                speckles = lp_filt(cp.real(speckles), "gaussian", 15)
                mask = cp.exp(1j * speckles * cp.pi * 1e3) * supp
                use_filter = False
            case "binary":
                supp = np.abs(lp_filt(supp, "gaussian", 1))
                mask = supp * speckles
                phase_radius = fftshift(
                    cp.exp(1.0j * (r_grid**2) / p.obj.camera_n / 1.5)
                )
                mask_binary_phase = ifftn(fftn(mask) * phase_radius)
                mask_binary_phase = (cp.angle(mask_binary_phase) > 0).astype(float)
                mask = ifftn(fftn(mask_binary_phase) * phase_radius.conj())
            case _:
                raise TypeError(
                    'p.MASK_PARAM["Type"] - your mask type is bad and you should feel bad'
                )
        if use_filter is True:
            mask = lp_filt(mask, p.mask.filter_shape, p.mask.filter_outer_limit)
            mask -= lp_filt(mask, p.mask.filter_shape, p.mask.filter_inner_limit)
        # mask *= cp.exp(1j * x_grid * .5 * k)
        masks[:, :, k] = mask
    return masks, supp_out


def load_objects(frames, p: Param) -> Tuple[np.ndarray, np.ndarray]:
    amp = Image.open("IN-subset/" + str(frames[0]) + ".jpg")
    phase = Image.open("IN-subset/" + str(frames[1]) + ".jpg")
    amp = cp.asarray(amp.resize((p.obj.supp_size,) * 2, Image.HAMMING))
    phase = cp.asarray(phase.resize((p.obj.supp_size,) * 2, Image.HAMMING))

    obj = cp.asarray(amp).astype(float) * cp.exp(1.0j * phase / phase.max() * 2 * cp.pi)

    pad = int((p.scenario.camera_n - p.obj.supp_size) / 2)
    obj = cp.pad(obj, ((pad,) * 2,) * 2)

    match p.obj.supp_shape:
        case "square":
            supp = rectangle(p.scenario.camera_n, (p.obj.supp_size - 5,) * 2)
        case "circ":
            supp = circle(p.scenario.camera_n, p.obj.supp_size / 2 - 8)
        case _:
            raise TypeError('p.obj.support_shape should be "square" or "circ"')
    supp = np.abs(lp_filt(supp, "gaussian", 3)) ** 2
    obj = supp * lp_filt(obj, p.obj.filter_shape, p.obj.filter_outer_limit)

    return obj, supp
