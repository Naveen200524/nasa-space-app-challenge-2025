"""
Noise masker for InSight using pressure (APSS) channel.
Functions:
- compute_pressure_gust_mask(pressure_array, sr, win_sec, std_thresh)
- dilate_mask(mask, sr, dilate_sec)
- apply_mask_to_seismic(seismic_data, mask, mode='zero'|'nan'|'clip')
- mask_seismic_with_pressure(seismic, pressure, sr, ...)
"""

import numpy as np
from scipy.ndimage import binary_dilation

def compute_pressure_gust_mask(pressure, sr, win_sec=1.0, std_thresh=3.0):
    """
    Sliding-window std-based gust detector on pressure time-series.

    Returns boolean mask array (True -> gust / contaminated).
    """
    p = np.asarray(pressure, dtype=np.float64)
    w = max(1, int(win_sec * sr))
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if w >= n:
        local_std = np.std(p)
        return np.full(n, local_std > 0.0, dtype=bool)

    # rolling mean and rolling squared sum to approximate variance
    cumsum = np.cumsum(np.insert(p, 0, 0.0))
    cumsum2 = np.cumsum(np.insert(p * p, 0, 0.0))
    means = (cumsum[w:] - cumsum[:-w]) / w
    means2 = (cumsum2[w:] - cumsum2[:-w]) / w
    var = np.maximum(means2 - means * means, 0.0)
    local_std_center = np.sqrt(var)

    # pad to original length
    pad_left = w // 2
    pad_right = n - (len(local_std_center) + pad_left)
    local_std = np.concatenate([
        np.full(pad_left, local_std_center[0] if len(local_std_center) else 0.0),
        local_std_center,
        np.full(pad_right, local_std_center[-1] if len(local_std_center) else 0.0)
    ])

    median_std = np.median(local_std + 1e-12)
    mask = local_std > (std_thresh * median_std)
    return mask.astype(bool)

def dilate_mask(mask, sr, dilate_sec=2.0):
    """
    Dilate mask by dilate_sec seconds on each side (binary dilation).
    """
    if dilate_sec <= 0:
        return mask
    d = max(1, int(dilate_sec * sr))
    structure = np.ones(d, dtype=bool)
    return binary_dilation(mask, structure=structure).astype(bool)

def apply_mask_to_seismic(seismic, mask, mode='zero'):
    """
    Apply boolean mask to seismic array.
    mode:
      - 'zero' : set masked samples to 0.0
      - 'nan'  : set masked samples to np.nan
      - 'clip' : attenuate masked samples (multiply by 0.3)
    """
    s = seismic.copy()
    if len(mask) != len(s):
        # try to broadcast / trim
        m = np.resize(mask, s.shape)
    else:
        m = mask
    if mode == 'zero':
        s[m] = 0.0
    elif mode == 'nan':
        s[m] = np.nan
    elif mode == 'clip':
        s[m] = s[m] * 0.3
    else:
        raise ValueError("mode must be 'zero'|'nan'|'clip'")
    return s

def mask_seismic_with_pressure(seismic, pressure, sr, win_sec=1.0, std_thresh=3.0, dilate_sec=2.0, mode='zero'):
    """
    Convenience function: returns (masked_seismic, mask_boolean_array)
    """
    mask = compute_pressure_gust_mask(pressure, sr, win_sec=win_sec, std_thresh=std_thresh)
    mask = dilate_mask(mask, sr, dilate_sec=dilate_sec)
    masked = apply_mask_to_seismic(seismic, mask, mode=mode)
    return masked, mask

