import numpy as np
from warnings import warn
import math
import torch.nn.functional as F
import torch
import torchvision
from scipy import spatial
import copy

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters


def blob_dog(
    image,
    th,
    min_sigma=1,
    max_sigma=50,
    sigma_ratio=1.6,
    threshold=0.5,
    overlap=0.5,
    *,
    threshold_rel=None,
    exclude_border=True,
    device="cuda",
):
    image = image.astype(np.float32)
    image = torch.from_numpy(image).to(device)
    scalar_sigma, min_sigma, max_sigma = _prep_sigmas(
        image.ndim,
        min_sigma,
        max_sigma,
    )

    if sigma_ratio <= 1.0:
        raise ValueError("sigma_ratio must be > 1.0")

    log_ratio = math.log(sigma_ratio)
    k = sum(
        math.log(max_s / min_s) / log_ratio + 1
        for max_s, min_s in zip(max_sigma, min_sigma)
    )
    k /= len(min_sigma)
    k = int(k)

    ratio_powers = tuple(sigma_ratio**i for i in range(k + 1))
    sigma_list = tuple(tuple(s * p for s in min_sigma) for p in ratio_powers)

    dog_image_cube = torch.empty(image.shape + (k,), dtype=torch.float, device=device)

    gaussian_previous = torchvision.transforms.functional.gaussian_blur(
        image[None, None],
        kernel_size=list(
            2 * np.round(4 * np.asarray(sigma_list[0])).astype(np.int32) + 1
        ),
        sigma=list(np.asarray(sigma_list[0]).astype(np.int32)),
    )[0, 0]
    for i, s in enumerate(sigma_list[1:]):
        gaussian_current = torchvision.transforms.functional.gaussian_blur(
            image[None, None],
            kernel_size=list(2 * np.round(4 * np.asarray(s)).astype(np.int32) + 1),
            sigma=list(np.asarray(s).astype(np.int32)),
        )[0, 0]

        dog_image_cube[..., i] = gaussian_previous - gaussian_current
        gaussian_previous = gaussian_current
    # normalization factor for consistency in DoG magnitude
    sf = 1 / (sigma_ratio - 1)
    dog_image_cube *= sf

    exclude_border = _format_exclude_border(image.ndim, exclude_border)
    local_maxima = peak_local_max(
        dog_image_cube,
        segment=image > th,
        threshold_abs=threshold,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
        footprint=np.ones((3,) * (image.ndim + 1)),
        min_distance=1,
    )

    # Catch no peaks
    if local_maxima.numel() == 0:
        return torch.empty((0, image.ndim + (1 if scalar_sigma else image.ndim))).to(
            device
        )

    # Convert local_maxima to float64
    lm = copy.deepcopy(local_maxima)  # .astype(np.float64)

    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigma_list = torch.from_numpy(np.asarray(sigma_list)).to(torch.float).to(device)
    sigmas_of_peaks = sigma_list[local_maxima[:, -1]]

    if scalar_sigma:
        # select one sigma column, keeping dimension
        sigmas_of_peaks = sigmas_of_peaks[:, 0:1]
    # Remove sigma index and replace with sigmas

    lm = torch.hstack([lm[:, :-1], sigmas_of_peaks])

    sigma_dim = sigmas_of_peaks.shape[1]
    return _prune_blobs(lm, overlap, sigma_dim=sigma_dim)


def _compute_disk_overlap(d, r1, r2):
    ratio1 = (d**2 + r1**2 - r2**2) / (2 * d * r1)
    ratio1 = torch.clip(ratio1, -1, 1)
    acos1 = torch.arccos(ratio1)

    ratio2 = (d**2 + r2**2 - r1**2) / (2 * d * r2)
    ratio2 = torch.clip(ratio2, -1, 1)
    acos2 = torch.arccos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = r1**2 * acos1 + r2**2 * acos2 - 0.5 * torch.sqrt(torch.abs(a * b * c * d))
    return area / (math.pi * (torch.minimum(r1, r2) ** 2))


def blob_overlap(blob1, blob2, *, sigma_dim=1):
    ndim = blob1.shape[-1] - sigma_dim
    if ndim > 3:
        return np.zeros((blob1.shape[0],))
    root_ndim = math.sqrt(ndim)
    blob1_mask = blob1[:, -1] > blob2[:, -1]
    blob2_mask = blob1[:, -1] <= blob2[:, -1]
    max_sigma = torch.ones((blob1.shape[0], 1)).to(blob1.device)

    max_sigma[blob1_mask] = blob1[:, -sigma_dim:][blob1_mask]
    max_sigma[blob2_mask] = blob2[:, -sigma_dim:][blob2_mask]
    r1 = torch.ones((blob1.shape[0])).to(blob1.device)
    r1[blob2_mask] = (blob1[:, -1] / blob2[:, -1])[blob2_mask]
    r2 = torch.ones((blob1.shape[0])).to(blob1.device)
    r2[blob1_mask] = (blob2[:, -1] / blob1[:, -1])[blob1_mask]

    pos1 = blob1[:, :ndim] / (max_sigma * root_ndim)
    pos2 = blob2[:, :ndim] / (max_sigma * root_ndim)

    d = torch.sqrt(torch.sum((pos2 - pos1) ** 2, dim=-1))

    output = _compute_disk_overlap(d, r1, r2)
    output[d > r1 + r2] = 0
    output[d <= abs(r1 - r2)] = 1
    output[(blob1[:, -1] == 0) * (blob2[:, -1] == 0)] = 0
    return output


def _prune_blobs(blobs_array, overlap, *, sigma_dim=1):
    sigma = blobs_array[:, -sigma_dim:].max()
    distance = 2 * sigma * math.sqrt(blobs_array.shape[1] - sigma_dim)

    tree = spatial.cKDTree(blobs_array[:, :-sigma_dim].cpu().data.numpy())

    pairs = torch.from_numpy(tree.query_pairs(distance, output_type="ndarray")).to(
        blobs_array.device
    )
    if len(pairs) == 0:
        return blobs_array
    else:
        blob1, blob2 = blobs_array[pairs[:, 0], :], blobs_array[pairs[:, 1], :]
        mask = blob_overlap(blob1, blob2, sigma_dim=sigma_dim) > overlap
        blobs_array[
            torch.unique(pairs[:, 0][mask * (blob2[:, -1] > blob1[:, -1])]), -1
        ] = 0
        blobs_array[
            torch.unique(pairs[:, 1][mask * (blob1[:, -1] > blob2[:, -1])]), -1
        ] = 0
    return blobs_array[blobs_array[:, -1] != 0, :]


def peak_local_max(
    image,
    segment=None,
    min_distance=1,
    threshold_abs=None,
    threshold_rel=None,
    exclude_border=True,
    num_peaks=np.inf,
    footprint=None,
    labels=None,
    num_peaks_per_label=np.inf,
    p_norm=np.inf,
):

    if (footprint is None or footprint.size == 1) and min_distance < 1:
        warn(
            "When min_distance < 1, peak_local_max acts as finding "
            "image > max(threshold_abs, threshold_rel * max(image)).",
            RuntimeWarning,
            stacklevel=2,
        )

    border_width = _get_excluded_border_width(image, min_distance, exclude_border)

    threshold = _get_threshold(image, threshold_abs, threshold_rel)

    if footprint is None:
        size = 2 * min_distance + 1
        footprint = np.ones((size,) * image.ndim, dtype=bool)

    mask = _get_peak_mask(image, segment, footprint, threshold)
    mask = _exclude_border(mask, border_width)

    # Select highest intensities (num_peaks)
    coordinates = _get_high_intensity_peaks(
        image, mask, num_peaks, min_distance, p_norm
    )

    return coordinates


def _get_high_intensity_peaks(image, mask, num_peaks, min_distance, p_norm):
    """
    Return the highest intensity peak coordinates.
    """
    # get coordinates of peaks
    coord = torch.nonzero(mask)
    intensities = image[coord[:, 0], coord[:, 1], coord[:, 2]]

    # Highest peak first
    idx_maxsort = torch.argsort(-intensities)
    coord = coord[idx_maxsort, :]

    # TODO: the code below is not used ... If no need any more, remove it.
    # if np.isfinite(num_peaks):
    #     max_out = int(num_peaks)
    # else:
    #     max_out = None

    if len(coord) > num_peaks:
        coord = coord[:num_peaks, :]

    return coord


def ensure_spacing(
    coords,
    spacing=10,
    p_norm=np.inf,
    min_split_size=50,
    max_out=None,
    *,
    max_split_size=2000,
):
    if len(coords):
        coords = torch.atleast_2d(coords)
        # coords = coords.cpu().data.numpy()
        if not np.isscalar(spacing):
            spacing = spacing.cpu().data.numpy()

        if min_split_size is None:
            batch_list = [coords]
        else:
            coord_count = coords.shape[0]
            split_idx = [min_split_size]
            split_size = min_split_size
            while coord_count - split_idx[-1] > max_split_size:
                split_size *= 2
                split_idx.append(split_idx[-1] + min(split_size, max_split_size))
            batch_list = torch.tensor_split(coords, split_idx)
        return torch.vstack(batch_list)


def _exclude_border(label, border_width):
    """Set label border values to 0."""
    # zero out label borders
    for i, width in enumerate(border_width):
        if width == 0:
            continue
        label[(slice(None),) * i + (slice(None, width),)] = 0
        label[(slice(None),) * i + (slice(-width, None),)] = 0
    return label


def _get_peak_mask(image, segment, footprint, threshold, mask=None):
    if footprint.size == 1 or image.size == 1:
        return image > threshold
    image_max = F.max_pool3d(
        image[None, None],
        kernel_size=footprint.shape,
        stride=1,
        padding=tuple(np.asarray(footprint.shape) // 2),
    )[0, 0]

    out = image == image_max

    # no peak for a trivial image
    image_is_trivial = torch.all(out) if mask is None else torch.all(out[mask])
    if image_is_trivial:  # synchronize
        out[:] = False
    image = torch.clip(image, torch.quantile(image, 0.1), torch.quantile(image, 0.9))
    threshold = filters.threshold_otsu(image[::2, ::2, :].cpu().data.numpy())
    out &= image > threshold
    return out * segment[..., None]


def _get_threshold(image, threshold_abs, threshold_rel):
    """Return the threshold value according to an absolute and a relative
    value.

    """
    threshold = threshold_abs if threshold_abs is not None else image.min()

    if threshold_rel is not None:
        threshold = max(threshold, threshold_rel * float(image.max()))
    # TODO: return host or device scalar?
    return float(threshold)


def _prep_sigmas(ndim, min_sigma, max_sigma):
    # if both min and max sigma are scalar, function returns only one sigma
    scalar_max = np.isscalar(max_sigma)
    scalar_min = np.isscalar(min_sigma)
    scalar_sigma = scalar_max and scalar_min

    # Gaussian filter requires that sequence-type sigmas have same
    # dimensionality as image. This broadcasts scalar kernels
    if scalar_max:
        max_sigma = (max_sigma,) * ndim
    if scalar_min:
        min_sigma = (min_sigma,) * ndim
    return scalar_sigma, min_sigma, max_sigma


def _format_exclude_border(img_ndim, exclude_border):
    if isinstance(exclude_border, tuple):
        if len(exclude_border) != img_ndim:
            raise ValueError(
                "`exclude_border` should have the same length as the "
                "dimensionality of the image."
            )
        for exclude in exclude_border:
            if not isinstance(exclude, int):
                raise ValueError(
                    "exclude border, when expressed as a tuple, must only "
                    "contain ints."
                )
        return exclude_border
    elif isinstance(exclude_border, int):
        return (exclude_border,) * img_ndim + (0,)
    elif exclude_border is True:
        raise ValueError("exclude_border cannot be True")
    elif exclude_border is False:
        return (0,) * (img_ndim + 1)
    else:
        raise ValueError(f"Unsupported value ({exclude_border}) for exclude_border")


def _get_excluded_border_width(image, min_distance, exclude_border):
    """Return border_width values relative to a min_distance if requested."""

    if isinstance(exclude_border, bool):
        border_width = (min_distance if exclude_border else 0,) * image.ndim
    elif isinstance(exclude_border, int):
        if exclude_border < 0:
            raise ValueError("`exclude_border` cannot be a negative value")
        border_width = (exclude_border,) * image.ndim
    elif isinstance(exclude_border, tuple):
        if len(exclude_border) != image.ndim:
            raise ValueError(
                "`exclude_border` should have the same length as the "
                "dimensionality of the image."
            )
        for exclude in exclude_border:
            if not isinstance(exclude, int):
                raise ValueError(
                    "`exclude_border`, when expressed as a tuple, must only "
                    "contain ints."
                )
            if exclude < 0:
                raise ValueError("`exclude_border` can not be a negative value")
        border_width = exclude_border
    else:
        raise TypeError(
            "`exclude_border` must be bool, int, or tuple with the same "
            "length as the dimensionality of the image."
        )

    return border_width
