import torch.nn as nn
import torch
import numpy as np
import math
import cv2
from skimage import morphology
import tqdm
import struct

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
import scipy
import copy
import pandas as pd
import skimage
import torch.nn.functional as F
from scipy import signal
import shutil
import gc
from lsfm_fuse.NSCT import NSCTdec, NSCTrec
import matplotlib.pyplot as plt
import ants
import tifffile
from skimage import measure
import numpy.ma as ma

import matplotlib.pyplot as plt
import numpy as np


def imagej_metadata_tags(metadata, byteorder):
    """Return IJMetadata and IJMetadataByteCounts tags from metadata dict.

    The tags can be passed to the TiffWriter.save function as extratags.

    """
    header = [{">": b"IJIJ", "<": b"JIJI"}[byteorder]]
    bytecounts = [0]
    body = []

    def writestring(data, byteorder):
        return data.encode("utf-16" + {">": "be", "<": "le"}[byteorder])

    def writedoubles(data, byteorder):
        return struct.pack(byteorder + ("d" * len(data)), *data)

    def writebytes(data, byteorder):
        return data.tobytes()

    metadata_types = (
        ("Info", b"info", 1, writestring),
        ("Labels", b"labl", None, writestring),
        ("Ranges", b"rang", 1, writedoubles),
        ("LUTs", b"luts", None, writebytes),
        ("Plot", b"plot", 1, writebytes),
        ("ROI", b"roi ", 1, writebytes),
        ("Overlays", b"over", None, writebytes),
    )

    for key, mtype, count, func in metadata_types:
        if key not in metadata:
            continue
        if byteorder == "<":
            mtype = mtype[::-1]
        values = metadata[key]
        if count is None:
            count = len(values)
        else:
            values = [values]
        header.append(mtype + struct.pack(byteorder + "I", count))
        for value in values:
            data = func(value, byteorder)
            body.append(data)
            bytecounts.append(len(data))

    body = b"".join(body)
    header = b"".join(header)
    data = header + body
    bytecounts[0] = len(header)
    bytecounts = struct.pack(byteorder + ("I" * len(bytecounts)), *bytecounts)
    return (
        (50839, "B", len(data), data, True),
        (50838, "I", len(bytecounts) // 4, bytecounts, True),
    )


def fusion_perslice(x, mask, GFr, device):
    n, c, m, n = x.shape
    GF = GuidedFilter(r=GFr, eps=1)
    x = torch.from_numpy(x).to(device)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).to(device).to(torch.float)

    result, num = GF(x, mask)
    num = num == (2 * GFr[1] + 1) * (2 * GFr[1] + 1) * GFr[0]
    result[num] = 1
    result = result / result.sum(0, keepdim=True)
    minn, maxx = x.min(), x.max()
    y_seg = x[:, c // 2 : c // 2 + 1, :, :] * result
    y = torch.clip(y_seg.sum(0), minn, maxx)

    return (
        y.squeeze().cpu().data.numpy().astype(np.uint16),
        result.squeeze().cpu().data.numpy(),
    )


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def diff_x(self, input, r):
        left = input[:, :, r : 2 * r + 1]
        middle = input[:, :, 2 * r + 1 :] - input[:, :, : -2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1 : -r - 1]
        output = torch.cat([left, middle, right], dim=2)
        return output

    def diff_y(self, input, r):
        left = input[:, :, :, r : 2 * r + 1]
        middle = input[:, :, :, 2 * r + 1 :] - input[:, :, :, : -2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1 : -r - 1]
        output = torch.cat([left, middle, right], dim=3)
        return output

    def forward(self, x):
        return self.diff_y(
            self.diff_x(x.sum(1, keepdims=True).cumsum(dim=2), self.r[1]).cumsum(dim=3),
            self.r[1],
        )


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()
        if isinstance(r, list):
            self.r = r
        else:
            self.r = [r, r]
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, x, y):
        mean_y_tmp = self.boxfilter(y)
        x, y = 0.001 * x, 0.001 * y
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()
        N = self.boxfilter(torch.ones_like(x))
        mean_x = self.boxfilter(x) / N
        mean_y = self.boxfilter(y) / N
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x
        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N
        return (
            mean_A * x[:, c_x // 2 : c_x // 2 + 1, :, :] + mean_b
        ) / 0.001, mean_y_tmp


def extendBoundary2(
    boundary,
    window_size,
):
    # window_size = window_size_list[1]
    # poly_order = poly_order_list[1]

    mask_1 = scipy.ndimage.binary_dilation(
        boundary > 0, structure=np.ones((1, window_size))
    )
    boundaryEM = copy.deepcopy(boundary)
    # plt.plot(boundary[28])
    for i in range(boundary.shape[0]):
        tmp = copy.deepcopy(boundary[i, :])
        p = np.where(tmp != 0)[0]
        if len(p) != 0:
            p0, p1 = p[0], p[-1]
            valid_slice = tmp[p0 : p1 + 1]
            g_left = [0] * window_size * 2 + [
                valid_slice[1] - valid_slice[0]
            ] * window_size * 2
            left_ext = np.cumsum(
                np.append(
                    -1
                    * signal.savgol_filter(g_left, window_size, 1)[
                        window_size
                        + window_size // 2
                        + 1 : 2 * window_size
                        + window_size // 2
                    ],
                    valid_slice[0],
                )[::-1]
            )[::-1]
            if p0 + 1 - len(left_ext) > 0:
                left_ext = np.pad(left_ext, [(p0 + 1 - len(left_ext), 0)], mode="edge")
            else:
                left_ext = left_ext[-(p0 + 1) :]
            boundary[i, : p0 + 1] = left_ext
            g_right = [valid_slice[-1] - valid_slice[-2]] * window_size * 2 + [
                0
            ] * window_size * 2
            right_ext = np.cumsum(
                np.concatenate(
                    (
                        np.array([valid_slice[-1]]),
                        signal.savgol_filter(g_right, window_size, 1)[
                            window_size
                            + window_size // 2
                            + 1 : 2 * window_size
                            + window_size // 2
                        ],
                    )
                )
            )
            if len(tmp[p1:]) - len(right_ext) > 0:
                right_ext = np.pad(
                    right_ext,
                    [(0, len(tmp[p1:]) - len(right_ext))],
                    mode="edge",
                )
            else:
                right_ext = right_ext[: len(tmp[p1:])]
            boundary[i, p1:] = right_ext

    return boundary


def extendBoundary(
    boundary,
    window_size_list,
    poly_order_list,
    spacing,
    _xy,
):
    boundaryEM = copy.deepcopy(boundary)
    if _xy == True:
        mask = morphology.binary_dilation(
            boundary != 0, np.ones((1, window_size_list[1]))
        )
        for dim in [1]:
            window_size = window_size_list[dim]
            poly_order = poly_order_list[dim]
            for i in range(boundary.shape[0]):
                p = np.where(boundary[i, :] != 0)[0]
                if len(p) != 0:
                    p0, p1 = p[0], p[-1]
                    boundary[i, :p0] = boundary[i, p0]
                    boundary[i, p1:] = boundary[i, p1]
                else:
                    boundary[i, :] = boundary[i, :]
    else:
        mask = boundary != 0

    boundary[~mask] = 0

    dist, ind = scipy.ndimage.distance_transform_edt(
        boundary == 0, return_distances=True, return_indices=True, sampling=spacing
    )
    boundary[boundary == 0] = boundary[ind[0], ind[1]][boundary == 0]

    return boundary


def EM2DPlus(
    segMask,
    f0,
    f1,
    window_size,
    poly_order,
    kernel2d,
    maxEpoch,
    device,
    _xy,
):

    def preComputePrior(seg, f0, f1):
        A = torch.cumsum(seg * f0, 0) + torch.flip(
            torch.cumsum(torch.flip(seg * f1, [0]), 0), [0]
        )
        return A

    def maskfor2d(segMask, window_size, m1, s1, n1):

        min_boundary = (
            torch.from_numpy(
                first_nonzero(
                    segMask,
                    None,
                    0,
                    segMask.shape[0] * 2,
                )
            )
            .to(torch.float)
            .to(device)
        )
        max_boundary = (
            torch.from_numpy(
                last_nonzero(
                    segMask,
                    None,
                    0,
                    -segMask.shape[0] * 2,
                )
            )
            .to(torch.float)
            .to(device)
        )
        if _xy == True:
            tmp = copy.deepcopy(
                min_boundary.cpu().data.numpy()
            )  # min_boundary.cpu().data.numpy()
            tmp[tmp != m1 * 2] = 0
            _, ind = scipy.ndimage.distance_transform_edt(
                tmp, return_distances=True, return_indices=True, sampling=[1, 1e3]
            )
            min_boundary[min_boundary == m1 * 2] = min_boundary[ind[0], ind[1]][
                min_boundary == m1 * 2
            ]
            tmp = copy.deepcopy(
                max_boundary.cpu().data.numpy()
            )  # max_boundary.cpu().data.numpy()
            tmp[tmp != -m1 * 2] = 0
            _, ind = scipy.ndimage.distance_transform_edt(
                tmp, return_distances=True, return_indices=True, sampling=[1, 1e3]
            )
            max_boundary[max_boundary == -m1 * 2] = max_boundary[ind[0], ind[1]][
                max_boundary == -m1 * 2
            ]

            mm = ~scipy.ndimage.binary_dilation(
                (segMask.sum(0) != 0).cpu().data.numpy(),
                np.ones((window_size[0] * 2 + 1, 1)),
            )
        else:
            mm = ~(segMask.sum(0) != 0).cpu().data.numpy()

        min_boundary[mm] = 0
        max_boundary[mm] = m1

        validMask = (segMask.sum(0) != 0).cpu().data.numpy().astype(np.float32)
        tmp = np.repeat(np.arange(s1)[:, None], n1, 1)
        maskrow = (tmp >= first_nonzero(validMask, None, axis=0, invalid_val=-1)) * (
            tmp <= last_nonzero(validMask, None, axis=0, invalid_val=-1)
        )
        tmp = np.repeat(np.arange(n1)[None, :], s1, 0)
        maskcol = (
            tmp >= first_nonzero(validMask, None, axis=1, invalid_val=-1)[:, None]
        ) * (tmp <= last_nonzero(validMask, None, axis=1, invalid_val=-1)[:, None])
        validMask[:] = copy.deepcopy(maskcol * maskrow).astype(np.float32)

        missingMask = (
            (segMask.sum(0) == 0).cpu().data.numpy()
            * (
                np.arange(n1)[None, :]
                >= first_nonzero(validMask, None, axis=1, invalid_val=-1)[:, None]
            )
            * (
                np.arange(n1)[None, :]
                <= last_nonzero(validMask, None, axis=1, invalid_val=-1)[:, None]
            )
        )
        validMask[missingMask] = 1

        valid_edge2 = skimage.util.view_as_windows(
            np.pad(
                validMask,
                (
                    (window_size[0] // 2, window_size[0] // 2),
                    (window_size[1] // 2, window_size[1] // 2),
                ),
                "constant",
                constant_values=0,
            ),
            window_size,
        ).sum((-2, -1))
        valid_edge2 = (
            (valid_edge2 < window_size[0] * window_size[1])
            * (valid_edge2 > 0)
            * validMask.astype(bool)
        )

        validMask_stripe = copy.deepcopy(validMask)
        validFor2D = skimage.util.view_as_windows(
            np.pad(
                validMask_stripe,
                (
                    (window_size[0] // 2, window_size[0] // 2),
                    (window_size[1] // 2, window_size[1] // 2),
                ),
                "constant",
                constant_values=0,
            ),
            window_size,
        ).sum((-2, -1))
        validFor2D = validFor2D == window_size[0] * window_size[1]
        validFor2D += valid_edge2

        return (
            torch.from_numpy(validFor2D).to(device),
            torch.arange(m1, device=device)[:, None, None],
            min_boundary,
            max_boundary,
        )

    def missingBoundary(boundaryTMP, s1, n1):
        boundaryTMP[boundaryTMP == 0] = np.nan
        a1 = np.isnan(boundaryTMP)
        boundaryTMP[np.isnan(boundaryTMP).sum(1) >= (n - 1), :] = 0
        for i in range(s1):
            boundaryTMP[i] = (
                pd.DataFrame(boundaryTMP[i])
                .interpolate("polynomial", order=1)
                .values[:, 0]
            )

        a2 = np.isnan(boundaryTMP)
        boundaryTMP[np.isnan(boundaryTMP)] = 0
        return boundaryTMP, (a2 == 0) * (a1)

    def selected_filter(x, validFor2D, min_boundary, max_boundary, kernel_high):
        w1, w2 = window_size
        dim0 = torch.zeros((1, n), dtype=torch.int).to(device)
        dim1 = torch.arange(0, n).to(device)
        y = torch.zeros_like(x)
        x_pad = F.pad(
            x[None, None], (w2 // 2, w2 // 2, w1 // 2, w1 // 2), mode="reflect"
        )[0, 0]
        min_boundary_pad = F.pad(
            min_boundary[None, None],
            (w2 // 2, w2 // 2, w1 // 2, w1 // 2),
            mode="reflect",
        )[0, 0]
        max_boundary_pad = F.pad(
            max_boundary[None, None],
            (w2 // 2, w2 // 2, w1 // 2, w1 // 2),
            mode="reflect",
        )[0, 0]
        for ind, i in enumerate(range(w1 // 2, s + w1 // 2)):
            xs = x_pad[i - w1 // 2 : i + w1 // 2 + 1, :]
            min_boundary_s = min_boundary_pad[i - w1 // 2 : i + w1 // 2 + 1, :]
            max_boundary_s = max_boundary_pad[i - w1 // 2 : i + w1 // 2 + 1, :]
            xs_unfold = xs.unfold(0, w1, 1).unfold(1, w2, 1)
            mask1 = (xs_unfold >= min_boundary[ind : ind + 1, :, None, None]) * (
                xs_unfold <= max_boundary[ind : ind + 1, :, None, None]
            )
            min_boundary_s_unfold = min_boundary_s.unfold(0, w1, 1).unfold(1, w2, 1)
            max_boundary_s_unfold = max_boundary_s.unfold(0, w1, 1).unfold(1, w2, 1)
            mask2 = (x[ind : ind + 1, :, None, None] >= min_boundary_s_unfold) * (
                x[ind : ind + 1, :, None, None] <= max_boundary_s_unfold
            )
            mask = mask1 * mask2
            mask[validFor2D[ind : ind + 1, :], :, :] = 1
            mask[:, :, w1 // 2, w2 // 2] = 1
            K = mask * kernel_high
            y[i - w1 // 2] = (K * xs_unfold).sum((-2, -1)) / K.sum((-2, -1))
            s_m = K.sum((-2, -1)) < 0.5
            if s_m.sum() > 0:
                xs_unfold_sort = torch.sort(
                    xs_unfold.reshape(1, n, -1), dim=-1, descending=True
                )[
                    0
                ]  # [:, :, :, 59//2-26:59//2+26+1]
                med_ind = (
                    xs_unfold_sort.shape[-1]
                    // 2
                    * torch.ones((1, n), dtype=torch.int).to(device)
                )
                median_result = xs_unfold_sort[dim0, dim1, med_ind]
                y[i - w1 // 2 : i - w1 // 2 + 1, :][s_m] = median_result[s_m]
        return y

    def init(x, validFor2D, bg_mask, min_boundary, max_boundary):
        dim0 = torch.zeros((1, n), dtype=torch.int).to(device)
        dim1 = torch.arange(0, n).to(device)
        w1, w2 = window_size
        y = torch.zeros_like(x)
        validFor2D0 = copy.deepcopy(validFor2D)

        x_pad_cpu = np.pad(x.cpu().data.numpy(), ((w2 // 2, w2 // 2)), mode="reflect")
        x_cpu = x.cpu().data.numpy()

        validFor2D = scipy.ndimage.binary_dilation(
            validFor2D.cpu().data.numpy(), np.ones((1, w2))
        )
        for i in range(s):
            t = torch.where(validFor2D0[i] > 0)[0]
            if len(t) > 0:
                a = t[0]
                b = t[-1]
                x_pad_cpu[i, a : b + w2] = np.pad(
                    x_cpu[i][a : b + 1], (w2 // 2, w2 // 2), mode="reflect"
                )
        x = torch.from_numpy(x_pad_cpu[:, w2 // 2 : -w2 // 2 + 1]).to(device)
        x_pad = F.pad(
            x[None, None], (w2 // 2, w2 // 2, w1 // 2, w1 // 2), mode="reflect"
        )[0, 0]

        validFor2D = torch.from_numpy(validFor2D).to(device)
        validFor2D = (
            F.pad(
                validFor2D[None, None] + 0.0,
                (w2 // 2, w2 // 2, w1 // 2, w1 // 2),
                mode="reflect",
            )[0, 0]
            > 0
        )
        for ind, i in enumerate(range(w1 // 2, s + w1 // 2)):
            xs = x_pad[i - w1 // 2 : i + w1 // 2 + 1, :]
            validFor2D_s = validFor2D[i - w1 // 2 : i + w1 // 2 + 1, :]
            xs_unfold = xs.unfold(0, w1, 1).unfold(1, w2, 1)  # .reshape(1, n, -1)
            validFor2D_unfold = validFor2D_s.unfold(0, w1, 1).unfold(1, w2, 1)
            mask = validFor2D_unfold
            mask[:, :, w1 // 2, w2 // 2] = 1
            med_ind = mask.sum((-2, -1)) // 2
            xs_unfold_sort = torch.sort(
                (mask * xs_unfold).reshape(1, n, -1), dim=-1, descending=True
            )[0]
            y[i - w1 // 2] = xs_unfold_sort[dim0, dim1, med_ind]
            y[i - w1 // 2][y[i - w1 // 2] == 0] = x[ind, :][y[i - w1 // 2] == 0]
        return y * validFor2D0

    m, s, n = segMask.shape
    segMask = segMask != 0
    bg_mask = segMask.sum(0) == 0
    feature = torch.zeros(m, s, n).to(device)

    for ss in range(s):
        feature[:, ss, :] = preComputePrior(
            segMask[:, ss, :],
            f0[:, ss, :],
            f1[:, ss, :],
        )

    cn = 0

    (
        validFor2D,
        coorMask,
        min_boundary,
        max_boundary,
    ) = maskfor2d(segMask, window_size, m, s, n)
    boundary0 = torch.argmax(feature, 0).cpu().data.numpy().astype(np.float32)

    tmp = np.arange(m)[:, None, None] > boundary0[None, :, :]

    boundary, _ = missingBoundary(copy.deepcopy(boundary0), s, n)
    if _xy:
        boundaryLS = (
            init(
                torch.from_numpy(boundary).to(device),
                validFor2D,
                bg_mask,
                min_boundary,
                max_boundary,
            )
            .cpu()
            .data.numpy()
        )
    else:
        boundaryLS = copy.deepcopy(boundary)

    boundaryLS = extendBoundary(
        boundaryLS,
        window_size,
        poly_order,
        [window_size[1] / window_size[0], 1.0],
        _xy=_xy,
    )
    tmp0 = np.arange(m)[:, None, None] > boundaryLS[None, :, :]
    boundary = torch.from_numpy(boundary).to(device)
    boundaryLS = torch.from_numpy(boundaryLS).to(device)
    boundaryOld = copy.deepcopy(boundary)

    boundaryLS = torch.maximum(boundaryLS, min_boundary)
    boundaryLS = torch.minimum(boundaryLS, max_boundary)

    w1, w2 = window_size
    for e in range(maxEpoch):
        Lambda = feature.max() / ((boundaryLS - boundary) ** 2 + 1).max()
        boundary[:] = torch.argmax(feature - Lambda * (boundaryLS - coorMask) ** 2, 0)
        changes = (
            100
            if e == 0
            else torch.quantile(torch.abs((boundaryOld - boundary) * (~bg_mask)), 0.99)
        )
        boundaryOld[:] = copy.deepcopy(boundary)
        boundaryLS[:] = selected_filter(
            boundary, bg_mask, min_boundary, max_boundary, kernel2d
        )

        cn = cn + 1 if changes < (5 if _xy else 2) else 0
        print(
            "\rNo.{:0>3d} iteration EM: maximum changes = {}".format(
                e, changes if e > 0 else "--"
            ),
            end="",
        )
    del feature, f0, f1

    boundaryLS = boundaryLS.cpu().data.numpy()

    return boundaryLS


def waterShed(xo, thresh, maxv, minv, m, n):
    x = np.zeros((m, n), dtype=np.float32)
    fg, bg = np.zeros((m, n), dtype=np.uint8), np.zeros((m, n), dtype=np.uint8)
    marker32, mm = np.zeros((m, n), dtype=np.int32), np.zeros((m, n), dtype=np.uint8)
    tmpMask = np.zeros((m, n), dtype=bool)
    if xo.max() > 0:
        x[:] = 255 * np.clip((xo - minv) / (maxv - minv), 0, 1)
    else:
        pass
    fg[:] = 255 * thresh.astype(bool)
    _, bg[:] = cv2.threshold(
        cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=10),
        1,
        128,
        1,
    )
    marker32[:] = cv2.watershed(
        cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_GRAY2BGR),
        np.int32(cv2.add(fg, bg)),
    )
    mm[:] = cv2.convertScaleAbs(marker32)
    tmpMask[:] = (mm != 128) * (mm != 1)

    tmpMask[::2, :] = skimage.morphology.binary_dilation(
        tmpMask[::2, :], np.ones((1, 3))
    )

    del thresh, x, fg, bg, marker32, mm
    return tmpMask


def refineShape(segMaskTop, segMaskBottom, topF, bottomF, s, m, n, r, _xy, max_seg):

    def missingBoundary(x, mask):
        data = copy.deepcopy(x) + 0.0
        data[mask] = np.nan
        data = pd.DataFrame(data).interpolate("polynomial", order=1).values[:, 0]
        return np.isinf(data) * (mask == 1)

    def outlierFilling(segMask):
        first = first_nonzero(segMask, None, 0, n - 1)
        last = last_nonzero(segMask, None, 0, 0)
        B = np.cumsum(segMask, 1)
        C = np.cumsum(segMask[:, ::-1], 1)[:, ::-1]
        D = (B > 0) * (C > 0) * (segMask == 0)
        D = D.astype(np.float32)
        E = copy.deepcopy(D)
        F = copy.deepcopy(D)
        b1 = (D[:, 1:] - D[:, :-1]) == 1
        b1 = np.concatenate((np.zeros((b1.shape[0], 1)), b1), 1).astype(np.uint8)

        b2 = (D[:, :-1] - D[:, 1:]) == 1
        b2 = np.concatenate((b2, np.zeros((b2.shape[0], 1))), 1).astype(np.uint8)

        b1 = (
            scipy.signal.convolve(b1, np.ones((11, 1)).astype(np.uint8), mode="same")
            == 11
        ) + 0.0  # 左
        b2 = (
            scipy.signal.convolve(b2, np.ones((11, 1)).astype(np.uint8), mode="same")
            == 11
        ) + 0.0  # 右

        for ii in range(b1.shape[0]):
            aa = np.where(b1[ii, :] == 1)[0]
            bb = np.where(b2[ii, :] == 1)[0]
            for a in aa:
                for b in bb:
                    if b >= a:
                        if (D[ii, a : b + 1] == 1).sum() == (b - a + 1):
                            E[ii, a : b + 1] = 0
                            break

        """
        aa_x, aa_y = np.where(b1 == 1)
        bb_x, bb_y = np.where(b2 == 1)
        if (len(aa_x)>0) and (len(bb_x)>0):
            same_row = aa_x[:, None] == bb_x[None, :]
            a_b_condition = aa_y[:, None] <= bb_y[None, :]
            b_minus_a_plus_1 = -aa_y[:, None]+bb_y[None, :]+1

            tmp = first_nonzero(None, same_row*a_b_condition, 1, np.nan)
            mask1 = ~np.isnan(tmp)
            if mask1.sum():
                aa_y = aa_y[mask1]
                bb_y = bb_y[tmp[mask1].astype(np.int32)]
                aa_x = aa_x[mask1]

                D = np.cumsum(D, 1)
                mask2 = (D[aa_x, bb_y+1] - D[aa_x, aa_y]) == bb_y-aa_y+1

                aa_y = aa_y[mask2]
                bb_y = bb_y[mask2]
                aa_x = aa_x[mask2]

                for ind, ii in enumerate(aa_x):
                    E[ii, aa_y[ind]:bb_y[ind]+1] = 0
        """
        Mask = (F - E) == 1  # ((-b1+b2) == -1)+((b1+b2) == 2)

        testa2 = measure.label(Mask, connectivity=2)
        props = measure.regionprops(testa2)
        A = np.zeros(Mask.shape, bool)

        for ind, p in enumerate(props):
            bbox = p.bbox
            if (bbox[3] - bbox[1]) / (bbox[2] - bbox[0]) < 2:
                first[bbox[1] : bbox[3]] = np.minimum(
                    np.linspace(first[bbox[1]], first[bbox[3]], bbox[3] - bbox[1]),
                    first[bbox[1] : bbox[3]],
                )
                last[bbox[1] : bbox[3]] = np.maximum(
                    np.linspace(last[bbox[1]], last[bbox[3]], bbox[3] - bbox[1]),
                    last[bbox[1] : bbox[3]],
                )
                mm = np.isin(testa2, ind + 1)
                mm[:, mm.sum(0) < r] = 0
                mm[:] = scipy.ndimage.binary_erosion(mm, np.ones((3, 1)))
                A[mm] = 1
        tmp = np.arange(segMask.shape[0])[:, None]
        result = (tmp >= first) * ((tmp <= last))
        result[A] = 1
        return segMask ^ result

    _mask, _maskl, _maskm = (
        np.zeros((s, m, n), dtype=bool),
        np.zeros((s, m, n), dtype=bool),
        np.zeros((s, m, n), dtype=bool),
    )

    for i in tqdm.tqdm(
        range(0, s), desc="refine pair-wise segmentation result: ", leave=False
    ):
        temp = np.linspace(0, m - 1, m, dtype=np.int32)[:, None]
        temp_top = segMaskTop[i, :, :]
        temp_bottom = segMaskBottom[i, :, :]

        boundaryCoordsBottom = last_nonzero(
            None, temp_bottom, axis=0, invalid_val=np.inf
        )
        boundaryCoordsTop = first_nonzero(None, temp_top, axis=0, invalid_val=np.inf)
        boundaryCoordsBottom2 = np.full(boundaryCoordsBottom.shape, np.inf)
        tmp = morphology.remove_small_objects(~np.isinf(boundaryCoordsBottom), r)
        boundaryCoordsBottom2[tmp] = boundaryCoordsBottom[tmp]
        boundaryCoordsTop2 = np.full(boundaryCoordsBottom.shape, np.inf)
        tmp = morphology.remove_small_objects(~np.isinf(boundaryCoordsTop), r)
        boundaryCoordsTop2[tmp] = boundaryCoordsTop[tmp]
        mask_bottom = np.isinf(boundaryCoordsBottom2) * (~np.isinf(boundaryCoordsTop2))
        mask_top = (~np.isinf(boundaryCoordsBottom2)) * np.isinf(boundaryCoordsTop2)
        mask_bottom_s = missingBoundary(boundaryCoordsBottom2, mask_bottom)
        mask_top_s = missingBoundary(boundaryCoordsTop2, mask_top)
        mask_bottom_l = mask_bottom_s ^ mask_bottom
        mask_top_l = mask_top_s ^ mask_top
        np.nan_to_num(boundaryCoordsBottom, copy=False, nan=0, posinf=0)
        np.nan_to_num(boundaryCoordsTop, copy=False, nan=m - 1, posinf=m - 1)

        segMask = (temp >= boundaryCoordsTop) * ((temp <= boundaryCoordsBottom))
        segMaskk = segMask.sum(0) > 0
        segMask += outlierFilling(segMask)
        segMask[:, mask_bottom_l] += temp_top[:, mask_bottom_l]
        segMask[:, mask_top_l] += temp_bottom[:, mask_top_l]
        _segMask = fillHole(segMask[None])[0]

        _maskl[i] = (
            _segMask  # scipy.ndimage.binary_erosion(_segMask, np.ones((re*2+1, re*2+1)))
        )

        if i < max(max_seg):
            f = first_nonzero(None, _segMask, axis=0, invalid_val=0)
            l = last_nonzero(None, _segMask, axis=0, invalid_val=m - 1)

            bottom_labeled = measure.label(temp_bottom, connectivity=2)
            top_labeled = measure.label(temp_top, connectivity=2)

            boundary_top_ind = top_labeled[f, np.arange(n)]
            boundary_bottom_ind = bottom_labeled[l, np.arange(n)]

            boundary_patch_bottom = bottom_labeled == boundary_bottom_ind[None, :]
            boundary_patch_top = top_labeled == boundary_top_ind[None, :]

            num_bottom = boundary_patch_bottom.sum(0)
            num_top = boundary_patch_top.sum(0)
            error_bottom = ((topF[i] > bottomF[i]) * boundary_patch_bottom).sum(0)
            error_top = ((topF[i] < bottomF[i]) * boundary_patch_top).sum(0)
            boundary_bottom_ind[
                ~(
                    (error_bottom < 11)
                    * (
                        num_bottom
                        / np.clip(((temp_bottom + temp_top) * _segMask).sum(0), 1, None)
                        < 0.2
                    )
                )
            ] = (boundary_bottom_ind.max() + 1)
            boundary_top_ind[
                ~(
                    (error_top < 11)
                    * (
                        num_top
                        / np.clip(((temp_bottom + temp_top) * _segMask).sum(0), 1, None)
                        < 0.2
                    )
                )
            ] = (boundary_top_ind.max() + 1)
            temp_bottom = (bottom_labeled == boundary_bottom_ind[None, :]) ^ temp_bottom
            temp_top = (top_labeled == boundary_top_ind[None, :]) ^ temp_top
            temp_bottom = morphology.remove_small_objects(temp_bottom, 121)
            temp_top = morphology.remove_small_objects(temp_top, 121)

            boundaryCoordsBottom = last_nonzero(
                None, temp_bottom, axis=0, invalid_val=0
            )
            boundaryCoordsTop = first_nonzero(None, temp_top, axis=0, invalid_val=m - 1)

            segMask = (temp >= boundaryCoordsTop) * ((temp <= boundaryCoordsBottom))
            mask = (segMask.sum(0) > 0) ^ segMaskk
            A = (boundaryCoordsBottom == 0) + (boundaryCoordsTop == (m - 1))
            boundaryCoordsBottom[A] = 0
            boundaryCoordsTop[A] = m - 1
            boundaryCoordsTop = boundaryCoordsTop[
                scipy.ndimage.distance_transform_edt(
                    boundaryCoordsTop == (m - 1),
                    return_distances=False,
                    return_indices=True,
                )
            ][
                0
            ]  # [(segMask.sum(0)>0)^segMaskk]
            boundaryCoordsBottom = boundaryCoordsBottom[
                scipy.ndimage.distance_transform_edt(
                    boundaryCoordsBottom == 0,
                    return_distances=False,
                    return_indices=True,
                )
            ][
                0
            ]  # [(segMask.sum(0)>0)^segMaskk]
            segMask[:, mask] = (
                (temp >= boundaryCoordsTop) * ((temp <= boundaryCoordsBottom))
            )[:, mask]
            segMask += outlierFilling(segMask)
            segMask[:, mask_bottom_l] += temp_top[:, mask_bottom_l]
            segMask[:, mask_top_l] += temp_bottom[:, mask_top_l]
            segMask = fillHole(segMask[None])[0] * _segMask

            _maskm[i] = (
                segMask  # scipy.ndimage.binary_erosion(segMask, np.ones((re*2+1, re*2+1)))
            )
        else:
            pass
    t = (np.arange(s)[:, None] >= np.array(max_seg)[None, :])[:, None, :].repeat(m, 1)
    _mask[t] = _maskl[t]
    _mask[~t] = _maskm[~t]
    if max(max_seg) > 0:
        for i in tqdm.tqdm(range(0, s), desc="refine pair-wise segmentation result: "):
            _mask[i] += outlierFilling(_mask[i])
    if _xy == False:
        return fillHole(_mask.transpose(1, 2, 0))
    else:
        _mask_small_tmp = _mask[:, :-1:2, :-1:2]

        _mask_small = np.zeros(
            (s, _mask_small_tmp.shape[1] * 2, _mask_small_tmp.shape[2] * 2), dtype=bool
        )
        _mask_small[:, ::2, ::2] = _mask_small_tmp

        with tqdm.tqdm(
            total=((_mask_small.shape[1] - 1) // 10 + 1)
            * ((_mask_small.shape[2] - 1) // 10 + 1),
            desc="refine along z: ",
            leave=False,
        ) as pbar:
            for i in range((_mask_small.shape[1] - 1) // 10 + 1):
                for j in range((_mask_small.shape[2] - 1) // 10 + 1):
                    _mask_small[:, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10] = (
                        morphology.remove_small_objects(
                            _mask_small[
                                :, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10
                            ],
                            5,
                        )
                    )
                    pbar.update(1)
        r = copy.deepcopy(_mask_small[:, ::2, ::2])
        _mask_small[:] = 1
        _mask_small[:, ::2, ::2] = r

        with tqdm.tqdm(
            total=((_mask_small.shape[1] - 1) // 10 + 1)
            * ((_mask_small.shape[2] - 1) // 10 + 1),
            desc="refine along z: ",
        ) as pbar:
            for i in range((_mask_small.shape[1] - 1) // 10 + 1):
                for j in range((_mask_small.shape[2] - 1) // 10 + 1):
                    _mask_small[:, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10] = (
                        morphology.remove_small_holes(
                            _mask_small[
                                :, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10
                            ],
                            5,
                        )
                    )
                    pbar.update(1)
        _mask[:, : _mask_small_tmp.shape[1] * 2, : _mask_small_tmp.shape[2] * 2] = (
            np.repeat(np.repeat(_mask_small[:, ::2, ::2], 2, 1), 2, 2)
        )

        _mask[:] = fillHole(_mask)
        return _mask


def fillHole(segMask):
    z, h, w = segMask.shape
    h += 2
    w += 2
    result = np.zeros(segMask.shape, dtype=bool)
    for i in range(z):
        _mask = np.pad(segMask[i], ((1, 1), (1, 1)))
        im_floodfill = 255 * (_mask.astype(np.uint8)).copy()
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(im_floodfill, mask, seedPoint=(0, 0), newVal=255)
        result[i, :, :] = (segMask[i] + (~im_floodfill)[1:-1, 1:-1]).astype(bool)
    return result


def sgolay2dkernel(window_size, order):
    n_terms = (order + 1) * (order + 2) / 2.0
    half_size = window_size // 2
    exps = []
    for row in range(order[0] + 1):
        for column in range(order[1] + 1):
            if (row + column) > max(*order):
                continue
            exps.append((row, column))
    indx = np.arange(-half_size[0], half_size[0] + 1, dtype=np.float64)
    indy = np.arange(-half_size[1], half_size[1] + 1, dtype=np.float64)
    dx = np.repeat(indx, window_size[1])
    dy = np.tile(indy, [window_size[0], 1]).reshape(
        window_size[0] * window_size[1],
    )
    A = np.empty((window_size[0] * window_size[1], len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1])
    return np.linalg.pinv(A)[0].reshape((window_size[0], -1))


def first_nonzero(arr, mask, axis, invalid_val=np.nan):
    if mask is None:
        mask = arr != 0
    if type(mask) is not np.ndarray:
        mask = mask.cpu().detach().numpy()
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(arr, mask, axis, invalid_val=np.nan):
    if mask is None:
        mask = arr != 0
    if type(mask) is not np.ndarray:
        mask = mask.cpu().detach().numpy()
    val = mask.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)
