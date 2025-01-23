import matplotlib.pyplot as plt
from plots import plot_rec_lines, scatter_rec_rec
from produce_om import load_masks, load_objects
import progressbar
from utils import align_phase, gpu2cpu, cpu2gpu, lp_filt, zero_dev
import numpy as np
from Param import Param
import dataclasses
from typing import Tuple, List

try:
    import cupy as cp
    from cupyx.scipy.fft import fftn, ifftn, fftshift
except ImportError as error:
    print("No cupy, using numpy instead")
    import numpy as cp
    from scipy.fft import fftn, ifftn, fftshift

from utils import expand_3d, norm_delta


def beam_blocker(intensity, p):
    bb = cp.ones_like(intensity)
    block_inds = slice(
        p.scenario.camera_n // 2 - p.scenario.block_width // 2,
        p.scenario.camera_n // 2 + p.scenario.block_width // 2,
    )
    bb[block_inds, :] = 0
    bb[:, block_inds] = 0
    bb = fftshift(bb)
    return intensity * bb, bb


def add_gauss_noise(noiseless_intensity, snr):
    if snr == cp.inf:
        noised_intensity = noiseless_intensity
    else:
        sig_db = 10 * cp.log10(noiseless_intensity.mean())
        noise_db = sig_db - snr
        noise_amp = 10 ** (noise_db / 20)  # STD
        noise = cp.random.normal(0, noise_amp, size=noiseless_intensity.shape)
    return (cp.sqrt(noiseless_intensity) + noise) ** 2


def power_spectrum(obj: cp.ndarray, masks: cp.ndarray, k: int):
    near_fields = expand_3d(obj, k) * masks
    far_fields = fftn(near_fields, axes=(0, 1))
    far_fields_intensity = cp.sum(abs(far_fields) ** 2, axis=2)
    return far_fields_intensity, far_fields, near_fields


def discretize(cont_intensity, dr):
    if dr == cp.inf:
        discrete_intensity = cont_intensity
    else:
        max_intensity = cont_intensity.max()
        max_gl = 2**dr - 1
        discrete_intensity = (
            cont_intensity / max_intensity * max_gl
        )  # set intensity between 0 and 2^dynamic_range
        discrete_intensity = cp.round(discrete_intensity) * max_intensity / max_gl
    return discrete_intensity


def create_measurements(obj, masks, p):
    far_fields_intensity = power_spectrum(obj, masks, p.scenario.mask_num)[0]
    intensity_noised = add_gauss_noise(far_fields_intensity, p.scenario.snr)
    intensity_disc = discretize(intensity_noised, p.scenario.bit_depth)
    intensity_measured, beam_block = beam_blocker(intensity_disc, p)

    delta_noise = norm_delta(np.sqrt(far_fields_intensity), np.sqrt(intensity_noised))
    delta_disc = norm_delta(np.sqrt(intensity_disc), np.sqrt(intensity_noised))
    delta_tot = norm_delta(np.sqrt(intensity_disc), np.sqrt(far_fields_intensity))

    if p.io.log_deltas is True:
        print(f"\n Noise Δ = {delta_noise:.2E}")
        print(f" Discretization Δ = {delta_disc:.2E}")
        print(f" Total Δ = {delta_tot:.2E}")

    return far_fields_intensity, intensity_measured, beam_block


def init_rec(p):
    rec = (
        cp.random.randn(*((p.obj.camera_n,) * 2))
        + cp.random.randn(*((p.obj.camera_n,) * 2)) * 1.0j
    ) * 1.5e3
    meas_err = cp.full(p.alg.iter_num, cp.inf)
    rec_err = cp.full(p.alg.iter_num, cp.inf)
    return rec, meas_err, rec_err


def mpie_multiplexed_cdi(
    meas: cp.ndarray,
    obj: cp.ndarray,
    mask_supp: cp.ndarray,
    masks: cp.ndarray,
    beam_block: cp.ndarray,
    rec0: cp.ndarray,  # Default is cp.inf to init internally
    p: Param,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    def momentum(velocity, recon, recon_mom, eta_mom):
        velocity = eta_mom * velocity + (recon - recon_mom)
        recon = recon + eta_mom * velocity
        recon_mom = recon.copy()
        return velocity, recon, recon_mom

    (iter_num, alpha, gamma, eta) = cpu2gpu(dataclasses.astuple(p.alg))
    m = p.scenario.mask_num
    rec, meas_err, rec_err = init_rec(p)
    rec = rec if (rec0 == cp.inf).any() else rec0
    rec = rec * mask_supp * masks[..., 0]
    for i in range(iter_num):
        rec_meas, far_fields, near_fields = power_spectrum(rec, masks, m)
        updated_far_fields = cp.where(
            expand_3d(beam_block, m) > 0,
            zero_dev(
                far_fields * np.sqrt(expand_3d(meas, m)),
                np.sqrt(expand_3d(rec_meas, m)),
            ),
            far_fields,
        )
        updated_near_fields = ifftn(updated_far_fields, axes=(0, 1))
        delta = ((updated_near_fields - near_fields) * masks.conj()) / (
            (1 - alpha) * cp.abs(masks) ** 2 + alpha * cp.abs(masks).max() ** 2
        )

        rec = rec + gamma * delta.sum(axis=2)
        rec = mask_supp * lp_filt(rec, p.obj.filter_shape, p.obj.filter_outer_limit)
        rec = align_phase(rec, obj)
        if i == 0:
            rec_mom, vel = rec.copy(), cp.zeros_like(rec)
        else:
            vel, rec, rec_mom = momentum(vel, rec, rec_mom, eta)
        meas_err[i] = norm_delta(np.sqrt(rec_meas) * beam_block, np.sqrt(meas))
        rec_err[i] = norm_delta(rec, obj)

        if (i % 100 == 0) and (i > 0) and meas_err[i] / meas_err[i - 100] > 0.9998:
            break
    return rec, meas_err, rec_err


def many_objects(
    p: Param,
    frames: np.ndarray = np.random.permutation(200),
    masks: cp.ndarray = cp.asarray(cp.inf),
    rec0: cp.ndarray = cp.asarray(cp.inf),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # same masks
    if (cp.inf == masks).any():
        masks, rec_supp = load_masks(p)
    else:
        rec_supp = load_masks(p)[1]
    recs, meas_errs, rec_errs = init_rec(p)
    objs, obj_supp = load_objects(frames, p)
    objs, recs, meas_errs, rec_errs = [
        cp.repeat(cp.expand_dims(x, 0), p.scenario.obj_num, 0)
        for x in [objs, recs, meas_errs, rec_errs]
    ]
    fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=100)

    # for i in range(p.scenario.obj_num):
    for i in progressbar.progressbar(range(p.scenario.obj_num)):
        objs[i], obj_supp = load_objects(frames[i * 2 : i * 2 + 2], p)
        _, meas, beam_block = create_measurements(objs[i], masks, p)
        (
            recs[i],
            meas_errs[i],
            rec_errs[i],
        ) = mpie_multiplexed_cdi(meas, objs[i], rec_supp, masks, beam_block, rec0, p)
        plot_rec_lines(gpu2cpu(rec_errs[[i]]), gpu2cpu(meas_errs[[i]]), ax=ax)
        display(fig, clear=True)  # TODO: change if not in jupyter
        plt.close(fig)
    return gpu2cpu((objs, recs, meas_errs, rec_errs, masks))


if __name__ == "__main__":
    pass
